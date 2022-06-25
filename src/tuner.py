import numpy as np
import pandas as pd
from sklearn import metrics

from src.optimizer import SMO


class Tuner(object):
    """
    Custom hyperparameter tuner for the
    SMO-based SVM classifier.
    """

    def __init__(self, C_range: list):
        self.optimizer = SMO
        self.hparam_space = {"C": C_range, "kernel": "linear", "tol": 1e-3}

    def perform(self, train_data: tuple, validation_data: tuple) -> None:
        """
        Performs the hyperparameter tuning.

        Args:
            train_data (tuple): pair of training design matrix and training labels
            validation_data (tuple): pair of validation design matrix and validation labels
        """
        x_train, y_train = train_data
        x_dev, y_dev = validation_data

        accuracy, precision, recall, f1_score = [], [], [], []

        for C in self.hparam_space["C"]:

            print(f"- training SMO-based classifier for C={C} (may take a while ...)")

            opt = self.optimizer(
                C=C, kernel=self.hparam_space["kernel"], tol=self.hparam_space["tol"]
            )
            _, b, w = opt.fit(x_train, y_train)

            preds = []
            if self.hparam_space["kernel"] == "linear":
                for i in range(len(x_dev)):
                    pred = (np.dot(w, x_dev[i].T) - b).sum()
                    preds.append(pred)

            preds = np.array(preds)
            y_pred = np.where(preds > 0, 1.0, -1.0)  # map to {-1, 1} depending on sign
            accuracy.append(metrics.accuracy_score(y_pred, y_dev))
            precision.append(metrics.precision_score(y_pred, y_dev))
            recall.append(metrics.recall_score(y_pred, y_dev))
            f1_score.append(metrics.f1_score(y_pred, y_dev))

        df = pd.DataFrame(
            {
                "C": self.hparam_space["C"],
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
            }
        )

        print(df)
