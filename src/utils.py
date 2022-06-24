import os

from sklearn.datasets import load_svmlight_file


def dir_is_empty(dirpath: str, exclude_dotfiles: bool = True) -> bool:

    contents = os.listdir(dirpath)

    if exclude_dotfiles:
        contents = [file for file in contents if not file.startswith(".")]

    return len(contents) == 0


def load_training_data() -> tuple:

    x, y = load_svmlight_file(os.path.join(os.getcwd(), "data/gisette_train.txt"))

    return x, y


def load_testing_data() -> tuple:

    x, y = load_svmlight_file(os.path.join(os.getcwd(), "data/gisette_test.txt"))

    return x, y
