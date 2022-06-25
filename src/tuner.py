from src.optimizer import SMO


class Tuner(object):
    def __init__(self, C_range: list):
        self.optimizer = SMO
        self.c_range = C_range

    def perform(self, train_data: tuple, validation_data: tuple) -> None:

        x_train, y_train = train_data
        x_dev, y_dev = validation_data

        for C in self.c_range:

            opt = self.optimizer(kernel="linear", C=C, tol=1e-3)
            alpha, b, w = opt.fit(x_train, y_train)

            # TODO: complete the logic
