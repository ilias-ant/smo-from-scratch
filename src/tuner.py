import copy
import json
import time
from datetime import timedelta

import pandas as pd
from sklearn import metrics

from src.optimizer import SMO


class Tuner(object):
    """
    Custom hyperparameter tuner for the SMO-based SVM classifier.

    Example:

        >>> from src import tuner
        >>>
        >>> X_train, y_train = ...
        >>> X_dev, y_dev = ...
        >>>
        >>> t = tuner.Tuner(C_range=[0.5, 1.0, 10.0])
        >>> t.perform(train_data=(X_train, y_train), validation_data=(X_dev, y_dev))
    """

    def __init__(self, C_range: list):
        self.optimizer = SMO
        self.hparam_space = {"C": C_range, "kernel": "linear", "tol": 1e-3}
        self.best_model_metric = "f1_score"

    def perform(self, train_data: tuple, validation_data: tuple) -> dict:
        """
        Performs hyperparameter tuning on C.

        Args:
            train_data (tuple): pair of training design matrix and training labels
            validation_data (tuple): pair of validation design matrix and validation labels

        Returns:
            The best set of hyperparameters.
        """
        x_train, y_train = train_data
        x_dev, y_dev = validation_data

        accuracy, precision, recall, f1_score, wall_time = [], [], [], [], []

        print(f"- hparam space to be explored:")
        print(json.dumps(self.hparam_space, indent=4, sort_keys=True))

        for C in self.hparam_space["C"]:

            print(f"- training SMO-based classifier for C={C} (may take a while ...)")

            opt = self.optimizer(
                C=C,
                kernel=self.hparam_space["kernel"],
                tol=self.hparam_space["tol"],
            )

            start = time.time()

            opt.fit(x_train, y_train)

            finish = time.time()

            y_pred = opt.predict(x_dev)
            y_pred = y_pred.reshape((-1, 1))
            accuracy.append(metrics.accuracy_score(y_pred, y_dev))
            precision.append(metrics.precision_score(y_pred, y_dev))
            recall.append(metrics.recall_score(y_pred, y_dev))
            f1_score.append(metrics.f1_score(y_pred, y_dev))
            wall_time.append(timedelta(seconds=round(finish - start)))

        print("- hyperparameter tuning completed.\n")

        df = pd.DataFrame(
            {
                "C": self.hparam_space["C"],
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "wall_time": wall_time,
            }
        )

        print(df)

        print(f"\n- returning best hparam set, according to {self.best_model_metric}.")

        idx = df[self.best_model_metric].idxmax()
        best_C = df["C"].iloc[idx]

        hparam_set = copy.deepcopy(self.hparam_space)
        hparam_set["C"] = best_C

        return hparam_set
