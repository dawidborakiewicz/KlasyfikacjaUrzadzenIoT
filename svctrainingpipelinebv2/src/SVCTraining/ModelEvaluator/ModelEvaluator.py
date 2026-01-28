import dataclasses
from datetime import datetime
import os
from typing import Dict, Any

from sklearn.metrics import make_scorer, recall_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import cross_validate, cross_val_score

@dataclasses.dataclass
class ValidationResult:
    f1: float
    passed: bool

class ModelEvaluator:
    def __init__(self,
                 output_dir: str = 'results',
                 cv_folds: int =5,
                 n_jobs: int=-1,
    ):
        self.output_dir = output_dir
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs


        os.makedirs(self.output_dir, exist_ok=True)

        self.scorers = {
            'macro_f1': 'f1_macro',
            'weighted_f1': 'f1_weighted',
            'accuracy': 'accuracy'
        }
        self.metric_functions = {
            "accuracy": lambda y, y_pred: accuracy_score(y, y_pred),
            "macro_f1": lambda y, y_pred: f1_score(y, y_pred, average="macro"),
            "weighted_f1": lambda y, y_pred: f1_score(y, y_pred, average="weighted"),
            "min_f1": lambda y, y_pred: f1_score(
                y, y_pred, average=None, zero_division=0
            ).min(),
            "min_recall": lambda y, y_pred: recall_score(
                y, y_pred, average=None, zero_division=0
            ).min(),
        }

    def evaluate_cv(self, model, X, y) -> dict[Any, tuple[Any, Any]]:
        cv_results = cross_validate(
            model,
            X,
            y,
            cv=self.cv_folds,
            scoring=self.scorers,
            n_jobs=self.n_jobs
        )

        return {
            metric.replace("test_", ""): (
                cv_results[metric].mean(),
                cv_results[metric].std()
            )
            for metric in cv_results
            if metric.startswith("test_")
        }

    def evaluate_test(self, model, X_test, y_test) -> Dict[str, float]:
        y_pred = model.predict(X_test)

        recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
            "min_f1": f1_per_class.min(),
            "min_recall": recall_per_class.min(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        }

    def save_report(
        self,
        experiment_name: str,
        best_params: dict,
        cv_results: Dict[str, tuple],
        test_results: Dict[str, float]
    ):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(
            self.output_dir,
            f"{experiment_name}_{timestamp}.txt"
        )

        with open(path, "w") as f:
            f.write(f"EXPERIMENT: {experiment_name}\n")
            f.write(f"Best params: {best_params}\n")
            f.write("=" * 80 + "\n\n")

            f.write("CROSS-VALIDATION RESULTS:\n")
            for k, (mean, std) in cv_results.items():
                f.write(f"{k}: {mean:.4f} (± {std:.4f})\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("TEST SET RESULTS:\n")

            for k, v in test_results.items():
                if k != "classification_report":
                    f.write(f"{k}: {v:.4f}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("CLASSIFICATION REPORT:\n")
            f.write(test_results["classification_report"])

    def validate(self, model, X, Y, threshold: float = 0.80) -> ValidationResult:
        Y_pred = model.predict(X)
        f1_macro = f1_score(Y, Y_pred, average="macro", zero_division=0)
        return ValidationResult(f1=f1_macro, passed=f1_macro >= threshold)