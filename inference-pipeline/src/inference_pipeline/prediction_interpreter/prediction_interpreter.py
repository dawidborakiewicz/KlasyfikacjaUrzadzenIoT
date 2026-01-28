import tempfile
import time
from dataclasses import dataclass
from typing import Optional, Any, List, Dict, Tuple

import mlflow
import numpy as np
import pandas as pd


@dataclass
class PredictionInterpreter:
    top_k: int = 1
    include_probabilities: bool = True
    unknown_label: str = "unknown"
    threshold: Optional[float] = None

    def _safe_predict_proba(self, model: Any, data: pd.DataFrame) -> Optional[np.ndarray]:
        proba_fn = getattr(model, "predict_proba", None)
        return proba_fn(data) if callable(proba_fn) else None

    def _class_names(self, model: Any) -> Optional[List[str]]:
        classes = getattr(model, "classes_", None)
        if classes is None:
            inner = getattr(model, "_model_impl", None)
            classes = getattr(inner, "classes_", None) if inner is not None else None
        return list(classes) if classes is not None else None

    def interpret(
        self,
        model: Any,
        data: pd.DataFrame,
        context: Optional[pd.DataFrame] = None,
    ) -> List[Dict]:
        """Zwraca listę rekordów z predykcjami (i opcjonalnie probabilistyką)."""
        y_pred = model.predict(data)
        y_proba = self._safe_predict_proba(model, data)
        class_names = self._class_names(model)

        results: List[Dict] = []
        for i in range(len(data)):
            row_ctx = (context.iloc[i].to_dict() if context is not None else {})
            rec: Dict[str, Any] = {
                "sample_idx": int(i),
                "pred_label": str(y_pred[i]),
                **row_ctx,
            }

            if self.include_probabilities and y_proba is not None:
                probs = y_proba[i]
                max_p = float(np.max(probs))
                if self.threshold is not None and max_p < self.threshold:
                    rec["pred_label"] = self.unknown_label
                rec["pred_confidence"] = max_p

                if self.top_k and self.top_k > 1:
                    idx_sorted = np.argsort(probs)[::-1][: self.top_k]
                    labels = [class_names[j] for j in idx_sorted] if class_names else [int(j) for j in idx_sorted]
                    rec["topk"] = [{"label": str(lbl), "prob": float(probs[j])} for lbl, j in zip(labels, idx_sorted)]

            results.append(rec)

        return results


def run_inference_and_log_to_mlflow(
    *,
    experiment_name: str = "DeviceInference",
    model: Any,
    model_name: str,                     
    data: pd.DataFrame,
    context: Optional[pd.DataFrame],
    interpreter: PredictionInterpreter,
    model_stage: str = "n/a",             
    model_version: str = "n/a",
    tracking_uri: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Zwraca:
    - predictions_df: per-wiersz wyniki (model_name, prediction, confidence, context...)
    - counts_df: rozkład predykcji (model_name, device_name, count)
    - metrics_df: 1-wierszowy DataFrame z metrykami (model_name, ...)
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    t0 = time.perf_counter()
    records = interpreter.interpret(model=model, data=data, context=context)
    infer_time = time.perf_counter() - t0

    df_results = pd.DataFrame(records)

    # jeśli brak predict_proba
    if "pred_confidence" not in df_results.columns:
        df_results["pred_confidence"] = np.nan

    # counts
    counts_df = (
        df_results.groupby("pred_label", dropna=False)
        .size()
        .reset_index(name="count")
        .rename(columns={"pred_label": "device_name"})
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


    predictions_df = df_results.copy()
    predictions_df = predictions_df.rename(columns={"pred_label": "prediction", "pred_confidence": "confidence"})

    # model_name wszędzie
    predictions_df.insert(0, "model_name", model_name)
    counts_df.insert(0, "model_name", model_name)

    rows_scored = int(len(predictions_df))
    avg_confidence = float(predictions_df["confidence"].mean()) if "confidence" in predictions_df.columns else float("nan")

    metrics_df = pd.DataFrame([{
        "model_name": model_name,
        "avg_confidence": avg_confidence,
        "rows_scored": rows_scored,
        "inference_time_sec": float(infer_time),
        "counts_rows": int(counts_df.shape[0]),
        "counts_cols": int(counts_df.shape[1]),
        "results_rows": int(predictions_df.shape[0]),
        "results_cols": int(predictions_df.shape[1]),
    }])


    with mlflow.start_run(run_name="inference"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_stage", model_stage)
        mlflow.log_param("model_version", str(model_version))

        mlflow.log_metric("avg_confidence", avg_confidence)
        mlflow.log_metric("rows_scored", rows_scored)
        mlflow.log_metric("inference_time_sec", float(infer_time))

        with tempfile.TemporaryDirectory() as tmpdir:
            counts_path = f"{tmpdir}/device_counts.csv"
            results_path = f"{tmpdir}/inference_results.csv"
            metrics_path = f"{tmpdir}/inference_metrics.csv"

            counts_df.to_csv(counts_path, index=False)
            predictions_df.to_csv(results_path, index=False)
            metrics_df.to_csv(metrics_path, index=False)

            mlflow.log_artifact(counts_path, artifact_path="inference_outputs")
            mlflow.log_artifact(results_path, artifact_path="inference_outputs")
            mlflow.log_artifact(metrics_path, artifact_path="inference_outputs")


    return predictions_df, counts_df, metrics_df