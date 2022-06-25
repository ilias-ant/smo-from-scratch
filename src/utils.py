import os

import numpy as np


def dir_is_empty(dirpath: str, exclude_dotfiles: bool = True) -> bool:

    contents = os.listdir(dirpath)

    if exclude_dotfiles:
        contents = [file for file in contents if not file.startswith(".")]

    return len(contents) == 0


def manual_load_training_data(max_lines: int = 5500) -> tuple:

    x, y, temp = [], [], []
    # open the file and store the lines
    with open(
        os.path.join(os.getcwd(), "data/gisette_scale"), "r", encoding="utf-8"
    ) as infile:
        lines = infile.read().split("\n")

    if max_lines is None or max_lines > len(lines):
        max_lines = len(lines)

    for line in lines[:max_lines]:

        if len(line):
            # split each line on the whitespace char
            observation = line.strip().split(" ")
            features = observation[1:]
            for feature in features:
                # store the features of each observation
                temp.append(float(feature.strip().split(":")[1]))
            if (len(temp)) != 4955:
                temp = []
                continue
            # append the features of the current observation
            x.append(temp)
            # append the class of the current observation
            y.append(float(observation[0]))
            temp = []

    return np.asmatrix(x), np.asarray(y)
