import atexit
import os
import signal
import subprocess
from datetime import datetime
from pathlib import Path

import data_preparation as dp
import inference_pipeline as ip
import RFTraining as rft
import SVCTraining as svct
import pandas as pd
import pcapflowexporter.PcapPacketExporter as ppe
from prefect import task, flow

#domyślny port Prefect to 4200
def start_prefect_server():
    proc = subprocess.Popen(
        ["prefect", "server", "start"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
    )

    os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"

    def cleanup():
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        except Exception:
            proc.kill()

    atexit.register(cleanup)
    return proc


@task(name="Pcap to csv")
def run_pcap_to_csv(src_filepath: str, learning: bool = False, labels_filepath: str = None):
    if learning:
        filename = f"learning_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        out_path = Path(f'CSVoutput/Learning/{filename}.csv')
    else:
        filename = f"inference_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        out_path = Path(f'CSVoutput/Inference/{filename}.csv')
    ppe.export_single_pcap_to_csv(src_filepath, out_path, labeling_enabled=learning, mapping_xlsx=labels_filepath)
    return out_path

@task(name="Data Preparation")
def data_preparation(src_filepath: str, balancing: bool = False):
    df = pd.read_csv(src_filepath)
    df = dp.handle_missing_values(df)
    df = dp.one_hot(df)
    if balancing:
        df = dp.balance_classes(df, label_column="label", method='value', multiplier=2.0)
    df_info, df_data = dp.split_columns(df)
    return df_info, df_data

@task(name="Inference")
def inference_pipeline(df_info: pd.DataFrame, df_data: pd.DataFrame, model_name: str, model):
    interpreter = ip.PredictionInterpreter()

    preds_df, dist_df, metrics_df = ip.run_inference_and_log_to_mlflow(
        model = model,
        model_name=model_name,
        data = df_data,
        context=df_info,
        interpreter = interpreter
    )
    metrics_df = metrics_df.drop(columns=["counts_rows","counts_cols", "results_rows", "results_cols"])
    return preds_df, dist_df, metrics_df


def validate_pcap_file(filepath: str) -> tuple[bool, str]:
    filepath = filepath.strip().strip('"').strip("'")

    if not os.path.exists(filepath):
        return False, f"Plik nie istnieje: {filepath}"

    if not os.path.isfile(filepath):
        return False, f"Podana ścieżka nie jest plikiem: {filepath}"

    if not (filepath.endswith(".pcap")):
        return False, "Plik musi mieć rozszerzenie .pcap"

    if not os.access(filepath, os.R_OK):
        return False, f"Brak uprawnień do odczytu pliku: {filepath}"

    if os.path.getsize(filepath) == 0:
        return False, "Plik jest pusty"

    return True, ""

def validate_xlsx_file(filepath: str) -> tuple[bool, str]:
    filepath = filepath.strip().strip('"').strip("'")

    if not os.path.exists(filepath):
        return False, f"Plik nie istnieje: {filepath}"

    if not os.path.isfile(filepath):
        return False, f"Podana ścieżka nie jest plikiem: {filepath}"

    if not (filepath.endswith(".xlsx")):
        return False, "Plik musi mieć rozszerzenie .xlsx"

    if not os.access(filepath, os.R_OK):
        return False, f"Brak uprawnień do odczytu pliku: {filepath}"

    if os.path.getsize(filepath) == 0:
        return False, "Plik jest pusty"

    return True, ""

@task(name="RandomForest model")
def random_forest_model(
    df_data: pd.DataFrame,
    label_column: str = "label",
    f1_threshold: float = 0.8,
    max_attempts: int = 1,
    base_seed: int = 42
):
    model, params, val_result, input_example, signature, model_name = rft.run_training_pipeline(
        df = df_data,
        label_column=label_column,
        f1_threshold=f1_threshold,
        max_attempts=max_attempts,
        base_seed=base_seed,
    )
    params_cols = {f"param_{k}": (v if isinstance(v, (int, float, str, bool)) else str(v))
                   for k, v in (params or {}).items()}

    result_df = pd.DataFrame([{
        "model_name": model_name,
        "test_f1_macro": float(val_result.f1),
        "passed_threshold": bool(val_result.passed),
        **params_cols
    }])
    return model_name, result_df

@task(name="SVC model")
def svc_model(
    df_data: pd.DataFrame,
    label_column: str = "label",
    f1_threshold: float = 0.8,
    max_attempts: int = 1,
    base_seed: int = 42
):
    model, params, val_result, input_example, signature, model_name = svct.run_training_pipeline(
        df = df_data,
        label_column=label_column,
        f1_threshold=f1_threshold,
        max_attempts=max_attempts,
        base_seed=base_seed,
    )
    params_cols = {f"param_{k}": (v if isinstance(v, (int, float, str, bool)) else str(v))
                   for k, v in (params or {}).items()}

    result_df = pd.DataFrame([{
        "model_name": model_name,
        "test_f1_macro": float(val_result.f1),
        "passed_threshold": bool(val_result.passed),
        **params_cols
    }])
    return model_name, result_df


@flow(name="Prediction from file", log_prints=True)
def inference_from_file_flow(filepath: str, model, model_name):
    if model is None:
        raise ValueError("Model musi zostać załadowany")
    print("Has predict_proba:", callable(getattr(model, "predict_proba", None)))
    print("Model type:", type(model))
    print("Has _model_impl:", hasattr(model, "_model_impl"))
    filepath = filepath.strip().strip('"').strip("'")
    is_valid, error_msg = validate_pcap_file(filepath)
    if not is_valid:
        raise ValueError(error_msg)

    csv_path = run_pcap_to_csv(filepath)
    file_info, file_data = data_preparation(csv_path)
    preds_df, dist_df, metrics_df = inference_pipeline(file_info, file_data, model_name=model_name, model=model)
    filename = f"inference_results_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_path = Path(f'CSVoutput/results/{filename}.csv')
    preds_df.to_csv(out_path, index=False)
    return f"Wnioskowanie zakończone. Wynik wnioskowania zapisany: {out_path}\n", metrics_df

@flow(name="Training RandomForest Model", log_prints=True)
def random_forest_training(data_filepath: str, labels_filepath: str):
    data_filepath = data_filepath.strip().strip('"').strip("'")
    labels_filepath = labels_filepath.strip().strip('"').strip("'")
    is_data_valid, data_error_msg = validate_pcap_file(data_filepath)
    if not is_data_valid:
        raise ValueError(data_error_msg)

    is_labels_valid, labels_error_msg = validate_pcap_file(data_filepath)
    if not is_labels_valid:
        raise ValueError(labels_error_msg)

    csv_path = run_pcap_to_csv(src_filepath=data_filepath,
                               learning=True,
                               labels_filepath=labels_filepath)
    file_info, file_data = data_preparation(src_filepath=csv_path, balancing=True)


    model_name, result_df = random_forest_model(df_data=file_data, base_seed=40)


    return f"Model {model_name} został poprawnie zarejestrowany", result_df

@flow(name="Training SVC Model", log_prints=True)
def svc_training(data_filepath: str, labels_filepath: str):
    data_filepath = data_filepath.strip().strip('"').strip("'")
    labels_filepath = labels_filepath.strip().strip('"').strip("'")
    is_data_valid, data_error_msg = validate_pcap_file(data_filepath)
    if not is_data_valid:
        raise ValueError(data_error_msg)

    is_labels_valid, labels_error_msg = validate_pcap_file(data_filepath)
    if not is_labels_valid:
        raise ValueError(labels_error_msg)

    csv_path = run_pcap_to_csv(src_filepath=data_filepath,
                               learning=True,
                               labels_filepath=labels_filepath)
    file_info, file_data = data_preparation(src_filepath=csv_path, balancing=True)


    model_name, result_df = svc_model(df_data=file_data, base_seed=40)


    return f"Model {model_name} został poprawnie zarejestrowany", result_df