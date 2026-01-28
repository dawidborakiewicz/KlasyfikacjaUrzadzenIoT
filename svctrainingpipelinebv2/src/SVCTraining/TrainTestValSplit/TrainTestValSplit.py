import pandas as pd
from sklearn.model_selection import train_test_split


class TrainTestValSplit:
    def __init__(self, min_data_rows: int = 100, random_state: int = 42):
        self.min_data_rows = min_data_rows
        self.random_state = random_state

    def split(
        self,
        data: pd.DataFrame,
        test_size: float = 0.15,
        val_size: float = 0.15,
        label_column: str = "device_class",
        shuffle: bool = True,
        stratify: bool = False,
    ):
        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1")
        if not (0 < val_size < 1):
            raise ValueError("val_size must be between 0 and 1")
        if (test_size + val_size) >= 1:
            raise ValueError("test_size and val_size cannot exceed 1")

        if data.shape[0] < self.min_data_rows:
            raise ValueError(
                f"The number of data rows is too small, should be at least {self.min_data_rows}"
            )

        if label_column not in data.columns:
            raise ValueError(f"Label column '{label_column}' not found in data")

        X = data.drop(label_column, axis=1)
        y = data[label_column]

        tmp_size = test_size + val_size
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X,
            y,
            test_size=tmp_size,
            random_state=self.random_state,
            shuffle=shuffle,
            stratify=y if stratify else None,
        )

        test_ratio = test_size / tmp_size
        X_test, X_val, y_test, y_val = train_test_split(
            X_tmp,
            y_tmp,
            test_size=(1 - test_ratio),  # część walidacyjna
            random_state=self.random_state,
            shuffle=shuffle,
            stratify=y_tmp if stratify else None,
        )

        return X_train, X_test, X_val, y_train, y_test, y_val
