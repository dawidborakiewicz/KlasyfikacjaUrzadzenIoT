import json
import time
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report

from ..HyperparameterTuner.HyperparameterTuner import HyperparameterTuner
from ..ModelEvaluator.ModelEvaluator import ModelEvaluator, ValidationResult
from ..SVCTrainer.SvcTrainer import SvcTrainer
from ..TrainTestValSplit.TrainTestValSplit import TrainTestValSplit


def _retry(fn, attempts=5, base_sleep=0.5):
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if i == attempts - 1:
                print(f"[MLflow] Giving up after {attempts} attempts: {e}")
                return
            sleep = base_sleep * (2 ** i)
            print(f"[MLflow] Error: {e}. Retrying in {sleep:.1f}s...")
            time.sleep(sleep)


def log_param_safe(k, v):
    _retry(lambda: mlflow.log_param(k, v))


def log_params_safe(d: dict):
    for k, v in (d or {}).items():
        log_param_safe(k, v)


def set_tag_safe(k, v):
    _retry(lambda: mlflow.set_tag(k, v))


def set_tags_safe(d: dict):
    for k, v in (d or {}).items():
        set_tag_safe(k, v)


def log_dict_safe(d: dict, artifact_file: str):
    _retry(lambda: mlflow.log_dict(d, artifact_file))


def log_metric_safe(k, v, step=None):
    _retry(lambda: mlflow.log_metric(k, v, step=step) if step is not None else mlflow.log_metric(k, v))


def log_param_space_as_artifact(param_space: dict, name="param_space.json"):
    import numpy as np

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

    serializable_param_space = convert_to_serializable(param_space)

    p = Path(name)
    p.write_text(json.dumps(serializable_param_space, ensure_ascii=False, indent=2), encoding="utf-8")
    _retry(lambda: mlflow.log_artifact(str(p), artifact_path="meta"))
    try:
        p.unlink()
    except Exception:
        pass


def run_training_pipeline(
    df: pd.DataFrame,
    label_column: str = "label",
    f1_threshold: float = 0.80,
    max_attempts: int = 1,
    base_seed: int = 42,
    ml_flow_tracking_uri: str = "http://127.0.0.1:8080",
    ml_flow_experiment_name: str = "SVC"
):
    start_time = time.time()

    def log_time(step_name: str):
        elapsed = time.time() - start_time
        print(f"[{elapsed:7.2f}s] {step_name}")
        mlflow.log_metric(f"time_{step_name.replace(' ', '_').lower()}", elapsed)

    mlflow.set_tracking_uri(ml_flow_tracking_uri)
    mlflow.set_experiment(ml_flow_experiment_name)

    with mlflow.start_run(run_name=f"TrainingRun_{datetime.now().strftime('%Y%m%d-%H%M%S')}") as parent_run:
        log_time("MLflow run started")
        log_params_safe({
            "f1_threshold_val": f1_threshold,
            "max_attempts": max_attempts
        })

        splitter = TrainTestValSplit(min_data_rows=100, random_state=base_seed)
        try:
            X_train, X_test, X_val, Y_train, Y_test, Y_val = splitter.split(
                df, test_size=0.15, val_size=0.15, label_column=label_column, shuffle=True, stratify=True
            )
        except ValueError as e:
            print(f"Error: {e}")
            return None

        log_time("Data split completed")
        log_params_safe({
            "train_data_shape": str(X_train.shape),
            "val_data_shape": str(X_val.shape),
            "test_data_shape": str(X_test.shape),
        })

        tuner = HyperparameterTuner()
        param_space = tuner.get_params()
        log_time("Hyperparameter space prepared")
        log_param_space_as_artifact(param_space)

        evaluator = ModelEvaluator()
        best_candidate = {"f1_val": -1.0, "params": None, "model": None, "run_id": None, "attempt": None}

        for attempt in range(1, max_attempts + 1):
            with mlflow.start_run(run_name=f"Attempt_{attempt}", nested=True) as child_run:
                current_seed = base_seed + attempt
                log_param_safe("random_state", current_seed)
                print("Child run ID:", child_run.info.run_id)

                trainer = SvcTrainer(
                    grid_verbose=3,
                    random_state=current_seed
                )

                print("Training with random state:", current_seed)
                trained = trainer.train(X_train, Y_train, param_space)

                best_model = trained[0]
                best_params = trained[1]

                log_time(f"Attempt {attempt} training finished")
                log_params_safe(best_params)

                val_res = evaluator.validate(best_model, X_val, Y_val, threshold=f1_threshold)

                y_val_pred = best_model.predict(X_val)
                val_recall_macro = recall_score(Y_val, y_val_pred, average="macro", zero_division=0)

                log_metric_safe("f1_val", val_res.f1, step=attempt)
                log_metric_safe("val_f1_macro", val_res.f1, step=attempt)
                log_metric_safe("val_recall_macro", val_recall_macro, step=attempt)

                if val_res.f1 > best_candidate["f1_val"]:
                    best_candidate["f1_val"] = val_res.f1
                    best_candidate["params"] = best_params
                    best_candidate["model"] = best_model
                    best_candidate["run_id"] = child_run.info.run_id
                    best_candidate["attempt"] = attempt
                    print(f"[{attempt}/{max_attempts}] New best model found! Val F1 macro: {val_res.f1:.4f}")

        log_time("All attempts completed")
        set_tag_safe("best_candidate_run_id", best_candidate["run_id"])

        best_model = best_candidate["model"]
        if best_model is None:
            print("Invalid state: No model was trained.")
            log_metric_safe("test_f1_macro", -1.0)
            return None

        y_test_pred = best_model.predict(X_test)

        test_f1_macro = f1_score(Y_test, y_test_pred, average="macro", zero_division=0)
        test_recall_macro = recall_score(Y_test, y_test_pred, average="macro", zero_division=0)
        test_precision_macro = precision_score(Y_test, y_test_pred, average="macro", zero_division=0)
        test_accuracy = accuracy_score(Y_test, y_test_pred)

        passed = test_f1_macro >= f1_threshold

        log_metric_safe("test_f1_macro", test_f1_macro)
        log_metric_safe("test_recall_macro", test_recall_macro)
        log_metric_safe("test_precision_macro", test_precision_macro)
        log_metric_safe("test_accuracy", test_accuracy)

        report = classification_report(Y_test, y_test_pred, output_dict=True, zero_division=0)
        log_dict_safe(report, "meta/classification_report_test.json")

        log_time("Final test evaluation done")

        input_example = X_train.iloc[:5] if isinstance(X_train, pd.DataFrame) else X_train[:5]
        signature = infer_signature(X_train, best_model.predict(X_train))

        set_tags_safe({
            "model_type": "SVC",
            "label_column": label_column,
            "passed_threshold": str(passed).lower(),
            "threshold_f1": f"{f1_threshold:.4f}",
            "test_f1_macro": f"{test_f1_macro:.4f}",
            "test_recall_macro": f"{test_recall_macro:.4f}",
            "best_val_f1_macro": f"{best_candidate['f1_val']:.4f}",
            "best_attempt": str(best_candidate.get("attempt")),
            "n_train": str(len(X_train)),
            "n_val": str(len(X_val)),
            "n_test": str(len(X_test)),
            "n_features": str(X_train.shape[1]) if hasattr(X_train, "shape") else "unknown",
        })

        artifact_path = "model"
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path=artifact_path,
            input_example=input_example,
            signature=signature,
        )

        registered_model_name = f"SVC_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        model_uri = f"runs:/{parent_run.info.run_id}/{artifact_path}"

        client = MlflowClient()
        mv = mlflow.register_model(model_uri, registered_model_name)

        client.set_model_version_tag(registered_model_name, mv.version, "passed_threshold", str(passed).lower())
        client.set_model_version_tag(registered_model_name, mv.version, "threshold_f1", f"{f1_threshold:.4f}")
        client.set_model_version_tag(registered_model_name, mv.version, "test_f1_macro", f"{test_f1_macro:.4f}")
        client.set_model_version_tag(registered_model_name, mv.version, "test_recall_macro", f"{test_recall_macro:.4f}")
        client.set_model_version_tag(registered_model_name, mv.version, "best_val_f1_macro", f"{best_candidate['f1_val']:.4f}")

        if passed:
            print(f"Model spełnił próg F1={f1_threshold}. Zarejestrowany jako {registered_model_name} v{mv.version}.")
        else:
            print(f"Model NIE spełnił progu, ale został zapisany i zarejestrowany jako {registered_model_name} v{mv.version} (passed_threshold=false).")

        return (
            best_candidate["model"],
            best_candidate["params"],
            ValidationResult(f1=test_f1_macro, passed=passed),
            input_example,
            signature,
            registered_model_name
        )
