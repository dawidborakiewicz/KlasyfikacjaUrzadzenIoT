import numpy as np


class HyperparameterTuner:
    def __init__(self, params=None):
        if params is None:
            self.params = {
                'svc__C': np.logspace(-2, 4, 13),
                'svc__gamma': np.logspace(-5, 1, 13),
            }
        else: self.params = params

    def get_params(self):
        return self.params