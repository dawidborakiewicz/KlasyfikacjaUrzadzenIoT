from typing import Dict, Iterable, Any, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib


class RandomForestTrainer:
    def __init__(
        self,
        random_state: int = 42,
        scoring: str = "f1_macro",
        rf_n_jobs: int = 1,
        cv_n_jobs: int = -1,
        class_weight: Optional[str] = "balanced",
        cv_splits: int = 5,
        n_iter: int = 40,
        verbose: int = 1,
        average: str = "macro",
    ):
        self.random_state = random_state
        self.scoring = scoring
        self.rf_n_jobs = rf_n_jobs
        self.cv_n_jobs = cv_n_jobs
        self.class_weight = class_weight
        self.cv_splits = cv_splits
        self.n_iter = n_iter
        self.verbose = verbose
        self.average = average

        self.search_: Optional[RandomizedSearchCV] = None
        self.model_: Optional[RandomForestClassifier] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_f1_: Optional[float] = None

    def train(
        self,
        X_train,
        y_train,
        param_distributions: Dict[str, Iterable[Any]],
    ) -> Tuple[Dict[str, Any], float]:
        print("Starting training with RandomizedSearchCV and StratifiedKFold")
        """RandomizedSearchCV + StratifiedKFold. Zwraca (best_params, best_f1)."""
        rf = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=self.rf_n_jobs,
            class_weight=self.class_weight,
            verbose=self.verbose,
        )
        print("Before StratifiedKFold")
        cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)

        print("before RandomizedSearchCV")
        search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_distributions,
            n_iter=self.n_iter,
            scoring=self.scoring,
            n_jobs=self.cv_n_jobs,
            cv=cv,
            verbose=self.verbose,
            random_state=self.random_state,
            refit=True, 
            return_train_score=False, 
        )
        with tqdm_joblib(tqdm(desc="RandomizedSearchCV progress", total=self.n_iter*self.cv_splits)) as progress_bar:
            search.fit(X_train, y_train)

        self.search_ = search
        self.model_ = search.best_estimator_
        self.best_params_ = search.best_params_

        return self.best_params_, self.search_.best_score_


    def get_model(self) -> RandomForestClassifier:
        if self.model_ is None:
            raise ValueError("Najpierw uruchom train().")
        return self.model_

    def get_features_importance(self) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Najpierw uruchom train().")
        return self.model_.feature_importances_



    def results_dataframe(self) -> pd.DataFrame:
        if self.search_ is None:
            raise ValueError("Najpierw uruchom train().")
        df = pd.DataFrame(self.search_.cv_results_)
        cols = [c for c in df.columns if c.startswith("param_")] + [
            "mean_test_score", "std_test_score", "rank_test_score"
        ]
        return df[cols].sort_values("rank_test_score").reset_index(drop=True)


