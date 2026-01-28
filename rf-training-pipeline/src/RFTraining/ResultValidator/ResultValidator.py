from dataclasses import dataclass
from sklearn.metrics import f1_score

@dataclass
class ValidationResult:
    f1: float
    passed: bool

class ResultValidator:
    def __init__(self, threshold: float, average: str = "macro"):
        self.threshold = threshold
        self.average = average

    def validate(self, model, X_val, y_val) -> ValidationResult:
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average=self.average)
        return ValidationResult(f1=f1, passed=f1 >= self.threshold)
