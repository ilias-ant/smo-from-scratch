import numpy as np
import pandas as pd
from sklearn import metrics

from src.optimizer import SMO


class Tuner(object):
    def __init__(self, C_range: list):
        self.optimizer = SMO
        self.hparam_space = {"C": C_range, "kernel": "linear", "tol": 1e-3}

    def perform(self, train_data: tuple, validation_data: tuple) -> None:

        x_train, y_train = train_data
        x_dev, y_dev = validation_data

        for C in self.hparam_space["C"]:

            u = []

            print(f"- training SMO-based classifier for C={C} (may take a while ...)")
            opt = self.optimizer(
                C=C, kernel=self.hparam_space["kernel"], tol=self.hparam_space["tol"]
            )
            alpha, b, w = opt.fit(x_train, y_train)

            if self.hparam_space["kernel"] == "linear":
                for i in range(len(x_dev)):
                    u.append(np.dot(w, x_dev[i].T) - b)

            df = pd.DataFrame(
                {
                    "C": self.hparam_space["C"],
                    "accuracy": metrics.accuracy_score(u, y_dev),
                    "precision": metrics.precision_score(u, y_dev),
                    "recall": metrics.recall_score(u, y_dev),
                    "f1_score": metrics.f1_score(u, y_dev),
                }
            )

            print(df)
