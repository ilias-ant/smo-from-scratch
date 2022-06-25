import os

import numpy as np


def dir_is_empty(dirpath: str, exclude_dotfiles: bool = True) -> bool:

    contents = os.listdir(dirpath)

    if exclude_dotfiles:
        contents = [file for file in contents if not file.startswith(".")]

    return len(contents) == 0


def load_data(filepath: str, max_lines: int = 6000) -> tuple:

    x, y, temp = [], [], []
    # open the file and store the lines
    with open(os.path.join(os.getcwd(), filepath), "r", encoding="utf-8") as infile:
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


def write_to_file(b: float, w: np.array) -> None:

    with open(os.path.join(os.getcwd(), "estimated_b.txt"), "w") as f:

        f.write(f"b: {b}\n")

    with open(os.path.join(os.getcwd(), "estimated_w.txt"), "w") as f:

        f.write("w:\n")
        for w_i in w:
            np.savetxt(f, w_i)
