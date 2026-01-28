import time
from typing import Iterable, Any, Dict, Optional

from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def f1_macro_scorer(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


class SvcTrainer:
    def __init__(
        self,
        scoring=make_scorer(f1_macro_scorer),
        kernel="rbf",
        class_weight="balanced",
        cache_size=1000,
        cv_n_splits=3,
        cv_shuffle=True,
        random_state=42,
        svc_verbose=False,
        grid_verbose=False,
        scaler=True,
        cv_refit=True,
        cv_n_jobs=-1,
    ):
        self.scoring = scoring
        self.kernel = kernel
        self.class_weight = class_weight
        self.cache_size = cache_size
        self.cv_n_splits = cv_n_splits
        self.cv_shuffle = cv_shuffle
        self.random_state = random_state
        self.svc_verbose = svc_verbose
        self.grid_verbose = grid_verbose
        self.scaler = scaler
        self.cv_refit = cv_refit
        self.cv_n_jobs = cv_n_jobs

        self.search_: Optional[GridSearchCV] = None
        self.model_: Optional[SVC] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.scoring_: Optional[float] = None

    def train(self, X_train, Y_train, param_grid: Dict[str, Iterable[Any]]):
        print("SVC Training with GridSearch starting...")

        steps = []
        if self.scaler:
            steps.append(("scaler", StandardScaler()))
        steps.append(
            (
                "svc",
                SVC(
                    kernel=self.kernel,
                    class_weight=self.class_weight,
                    random_state=self.random_state,
                    cache_size=self.cache_size,
                    verbose=self.svc_verbose,
                    probability=True
                ),
            )
        )
        pipeline = Pipeline(steps)

        cv = StratifiedKFold(
            n_splits=self.cv_n_splits,
            shuffle=self.cv_shuffle,
            random_state=self.random_state,
        )

        # (opcjonalnie) tylko informacyjnie
        n_candidates = 1
        if "svc__C" in param_grid:
            n_candidates *= len(param_grid["svc__C"])
        if "svc__gamma" in param_grid:
            n_candidates *= len(param_grid["svc__gamma"])
        print(f"Liczba kombinacji: {n_candidates}")

        print("Starting GridSearch WITHOUT subsampling")
        start = time.time()

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=self.scoring,
            cv=cv,
            refit=self.cv_refit,
            n_jobs=self.cv_n_jobs,
            verbose=self.grid_verbose,
        )

        # POPRAWKA: .values (bez nawiasów) jeśli pandas; jeśli numpy to zadziała też bez
        X_fit = X_train.values if hasattr(X_train, "values") else X_train
        y_fit = Y_train.values if hasattr(Y_train, "values") else Y_train

        search.fit(X_fit, y_fit)

        elapsed = time.time() - start
        print(f"GridSearch ended in {elapsed / 60:.2f} minutes")

        best_model = search.best_estimator_
        print(f"Best estimator: {best_model}")
        print(f"Best CV score: {search.best_score_:.6f}")

        self.search_ = search
        self.model_ = best_model
        self.best_params_ = search.best_params_
        self.scoring_ = search.best_score_

        return self.model_, self.best_params_, self.scoring_

    def get_model(self) -> SVC:
        if self.model_ is None:
            raise ValueError("Train model first!")
        return self.model_

    def get_scoring(self) -> float:
        if self.scoring_ is None:
            raise ValueError("Train scoring first!")
        return self.scoring_

    def get_best_params(self) -> Dict[str, Any]:
        if self.best_params_ is None:
            raise ValueError("Train best_params first!")
        return self.best_params_
