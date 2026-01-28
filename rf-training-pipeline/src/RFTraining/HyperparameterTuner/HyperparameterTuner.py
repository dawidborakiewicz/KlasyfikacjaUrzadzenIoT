

class HyperparameterTuner:
    def __init__(self, params=None, random_state=42):
        self.params = {
            "n_estimators": [x for x in range(10, 1500, 50)],
            "max_depth": [None] + [x for x in range(5, 30, 5)],
            "max_features": ["sqrt","log2"], 
            "min_samples_split": [x for x in range(2,5000, 250)],
            "min_samples_leaf": [x for x in range(1,101, 2)],
            "max_samples": [None] + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "criterion": ["gini"], 
        }
    def get_params(self):
        return self.params



