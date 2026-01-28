import gradio as gr
import time

from flows.file_inference_workflow import inference_from_file_flow, start_prefect_server, random_forest_training, svc_training
from mlflow_server.mlflow_setup import start_mlflow_server
from mlflow_server.mlflow_utils import *


def toggle_realtime_inference(is_running):
    if not is_running:
        return (
            True,  # nowy stan
            gr.update(value="⛔ Zatrzymaj wnioskowanie", variant="stop"),
            "⏳ Wnioskowanie w toku..."
        )
    else:
        return (
            False,
            gr.update(value="Uruchom wnioskowanie w czasie rzeczywistym", variant="primary"),
            "⛔ Wnioskowanie zatrzymane."
        )


EMPTY_DF = pd.DataFrame()

def run_training_router(data_filepath, labels_filepath, model_choice):
    if model_choice == "Random Forest":
        return run_RF_training(data_filepath, labels_filepath)
    elif model_choice == "SVC":
        return run_SVC_training(data_filepath, labels_filepath)
    else:
        return "⚠️ Nieznany wybór modelu.", EMPTY_DF

def run_file_inference(filepath, model, model_name, history_df):
    try:
        result, metrics_df = inference_from_file_flow(filepath, model=model, model_name=model_name)
        if history_df is None or not isinstance(history_df, pd.DataFrame):
            history_df = pd.DataFrame()

        if metrics_df is None or not isinstance(metrics_df, pd.DataFrame) or metrics_df.empty:
            return (result or "✅ Flow zakończony pomyślnie"), history_df, history_df

        metrics_df = metrics_df.copy()
        metrics_df.insert(0, "ts", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

        new_history = pd.concat([history_df, metrics_df], ignore_index=True)

        return (result or "✅ Flow zakończony pomyślnie"), new_history, new_history

    except ValueError as e:
        return f"❌ Błąd walidacji: {e}", EMPTY_DF, history_df
    except FileNotFoundError as e:
        return f"❌ Nie znaleziono pliku: {e}", EMPTY_DF, history_df
    except PermissionError as e:
        return f"❌ Brak uprawnień: {e}", EMPTY_DF, history_df
    except Exception as e:
        return f"❌ Nieoczekiwany błąd: {e}", EMPTY_DF, history_df

def run_RF_training(data_filepath, labels_filepath):
    try:
        result, results_df =  random_forest_training(data_filepath, labels_filepath)

        return (result or "✅ Flow zakończony pomyślnie"), results_df

    except ValueError as e:
        return f"❌ Błąd walidacji: {e}", EMPTY_DF
    except FileNotFoundError as e:
        return f"❌ Nie znaleziono pliku: {e}", EMPTY_DF
    except PermissionError as e:
        return f"❌ Brak uprawnień: {e}", EMPTY_DF
    except Exception as e:
        return f"❌ Nieoczekiwany błąd: {e}", EMPTY_DF

def run_SVC_training(data_filepath, labels_filepath):
    try:
        result, results_df =  svc_training(data_filepath, labels_filepath)

        return (result or "✅ Flow zakończony pomyślnie"), results_df

    except ValueError as e:
        return f"❌ Błąd walidacji: {e}", EMPTY_DF
    except FileNotFoundError as e:
        return f"❌ Nie znaleziono pliku: {e}", EMPTY_DF
    except PermissionError as e:
        return f"❌ Brak uprawnień: {e}", EMPTY_DF
    except Exception as e:
        return f"❌ Nieoczekiwany błąd: {e}", EMPTY_DF


def process_realtime_inference():
    time.sleep(3)
    return "✅ Wnioskowanie w czasie rzeczywistym zakończone (mock)."


def refresh_models():
    df, names = fetch_models_registry()
    return (
        gr.update(value=df),
        gr.update(choices=names, value=(names[0] if names else None)),
        "✅ Lista modeli odświeżona."
    )

with gr.Blocks(title="Klasyfikacja urządzeń IoT") as demo:
    is_running = gr.State(False)
    gr.Markdown("# 🔍 Klasyfikacja urządzeń IoT")

    with gr.Tabs() as tabs:
        with gr.TabItem("🏠 Strona główna", id=0):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Wybierz opcję:")
                    btn_inference = gr.Button("🚀 Start Program", variant="primary")
                    btn_training = gr.Button("🏋️‍♂️ Trenowanie Modeli", variant="secondary")

            btn_inference.click(
                fn=lambda: gr.update(selected=1),
                inputs=None,
                outputs=tabs
            )
            btn_training.click(
                fn=lambda: gr.update(selected=2),
                inputs=None,
                outputs=tabs
            )

        with gr.TabItem("🚀 Inferencja", id=1):
            gr.Markdown("## Pipeline klasyfikacji")

            with gr.Row():
                with gr.Column():
                    inference_status = gr.Markdown("")
                    inference_preview = gr.Dataframe(label="Metryki wnioskowań", interactive=False, wrap=True)
                    btn_clear = gr.Button("Wyczyść metryki", variant="secondary")
                    inference_history = gr.State(pd.DataFrame())
                with gr.Column():
                    model_status = gr.Markdown("")
                    model_preview = gr.Dataframe(
                        label="Zarejestrowane modele + tagi",
                        interactive=False,
                        wrap=True
                    )
                    model_choice = gr.Dropdown(
                        choices=[],
                        label="Wybierz model (MLflow Registry)",
                        interactive=True
                    )
                    with gr.Row():
                        refresh_btn = gr.Button("Odśwież modele", variant="secondary")
                        load_btn = gr.Button("Załaduj model", variant="primary")
                    loaded_model = gr.State(None)
                    loaded_model_name = gr.State("")

            with gr.Row():
                with gr.Column():
                    btn_start = gr.Button(
                        "Uruchom wnioskowanie w czasie rzeczywistym",
                        variant="primary",
                    )
                with gr.Column():
                    inference_file = gr.Textbox(
                        label="Ścieżka do pliku .pcap",
                        placeholder="/ścieżka/do/pliku.pcap",
                        info="Podaj pełną ścieżkę do lokalnego pliku .pcap. Ctrl+Shift+C kopiuje ścieżkę do zaznaczonego pliku.",
                        lines=1
                    )
                    btn_start_file = gr.Button(
                        "Uruchom wnioskowanie z pliku",
                        variant="secondary",
                        interactive=False  # Domyślnie wyłączony
                    )

            btn_back1 = gr.Button("🔙 Powrót do głównej")
            btn_back1.click(
                fn=lambda: gr.update(selected=0),
                inputs=None,
                outputs=tabs
            )

            btn_start.click(
                fn=toggle_realtime_inference,
                inputs=[is_running],
                outputs=[is_running, btn_start, inference_status]
            )

            btn_start_file.click(
                fn=run_file_inference,
                inputs=[inference_file, loaded_model, loaded_model_name, inference_history],
                outputs=[inference_status, inference_preview, inference_history],
            )

            load_btn.click(
                fn=load_selected_model,
                inputs=[model_choice],
                outputs=[loaded_model, loaded_model_name, model_status]
            )


            def clear_history():
                df = pd.DataFrame()
                return df, df


            btn_clear.click(
                fn=clear_history,
                inputs=None,
                outputs=[inference_preview, inference_history],
            )

            refresh_btn.click(
                fn=refresh_models,
                inputs=None,
                outputs=[model_preview, model_choice, model_status]
            )


            def toggle_button(text):
                return gr.update(interactive=bool(text and text.strip()))


            inference_file.change(
                fn=toggle_button,
                inputs=inference_file,
                outputs=btn_start_file
            )

        with gr.TabItem("️‍🏋️‍♂️ Trenowanie", id=2):
            gr.Markdown("## Trenowanie modelu")

            train_status = gr.Markdown("")
            with gr.Column():
                train_file= gr.Textbox(
                        label="Ścieżka do pliku .pcap",
                        placeholder="/ścieżka/do/pliku.pcap",
                        info="Podaj pełną ścieżkę do lokalnego pliku .pcap. Ctrl+Shift+C kopiuje ścieżkę do zaznaczonego pliku.",
                        lines=1
                    )
                labels_file = gr.Textbox(
                    label="Ścieżka do pliku .xlsx",
                    placeholder="/ścieżka/do/pliku.xlsx",
                    info="Podaj pełną ścieżkę do lokalnego pliku .xlsx. Ctrl+Shift+C kopiuje ścieżkę do zaznaczonego pliku.",
                    lines=1
                )
                training_results = gr.DataFrame(
                    label="Zarejestrowany model",
                    interactive=False,
                    wrap=True
                )
                model_choice_train = gr.Radio(
                    choices=["Random Forest", "SVC"],
                    value="Random Forest",
                    label="Wybierz model",
                    interactive=True,
                )
                btn_train = gr.Button("Rozpocznij trenowanie", variant="primary", interactive=False)

            btn_back2 = gr.Button("🔙 Powrót do głównej")
            btn_back2.click(
                fn=lambda: gr.update(selected=0),
                inputs=None,
                outputs=tabs
            )


            def update_button(text1, text2):
                return gr.update(interactive=bool(text1 and text1.strip() and text2 and text2.strip()))

            btn_train.click(
                fn=run_training_router,
                inputs=[train_file, labels_file, model_choice_train],
                outputs=[train_status, training_results]
            )

            train_file.change(
                fn=update_button,
                inputs=[train_file, labels_file],
                outputs=btn_train,
            )

            labels_file.change(
                fn=update_button,
                inputs=[train_file, labels_file],
                outputs=btn_train,
            )

    tabs.select(
        fn=lambda tab_id: refresh_models() if tab_id == 1 else (None, None, None),
        inputs=tabs,
        outputs=[model_preview, model_choice, model_status]
    )

prefect_proc = start_prefect_server()
mlflow_proc = start_mlflow_server(host="127.0.0.1", port=8080)
demo.queue()
demo.launch()