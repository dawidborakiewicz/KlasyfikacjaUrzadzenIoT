import pandas as pd
from sklearn.model_selection import train_test_split

class TrainTestValSplit:
    def __init__(self, min_data_rows: int =100, random_state: int =42, shuffle: bool =True, stratify: bool =False):
        self.min_data_rows = min_data_rows
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify = stratify

    def split(self, data: pd.DataFrame, test_size: float =0.15, val_size: float =0.15, label_column: str = 'target', shuffle: bool = None, stratify: bool = None):

        if shuffle is None:
            shuffle = self.shuffle
        if stratify is None:
            stratify = self.stratify


        if not (0 < test_size < 1) or not (0 < val_size < 1):
            raise ValueError("Ratios must be between 0 and 1")
        if test_size + val_size >= 1:
            raise ValueError("Sum of test_ratio and val_ratio must be less than 1")
        train_ratio = 1 - (test_size + val_size)
        if abs((train_ratio + test_size + val_size) - 1.0) > 1e-9:
            raise ValueError("Ratios must sum to 1.0")
        if data.shape[0] < self.min_data_rows:
            raise ValueError(f"Data must have at least {self.min_data_rows} rows, as defined in the class constructor.")
        x = data.drop(columns=[label_column])
        y = data[label_column]
        temp_val_and_test_size = test_size + val_size
        x_train, x_tmp, y_train, y_tmp = train_test_split(x, y, test_size=temp_val_and_test_size, random_state=self.random_state, shuffle=shuffle,stratify=y if stratify else None)
        x_test, x_val, y_test, y_val = train_test_split(x_tmp, y_tmp, test_size=val_size / temp_val_and_test_size, random_state=self.random_state, shuffle=shuffle,stratify=y_tmp if stratify else None)
        return x_train, y_train, x_test, y_test, x_val, y_val




