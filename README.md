# smo-from-scratch

[![python: supported versions](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Implementing Sequential Minimal Optimization algorithm from [John C. Platt's 1998 paper](https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/).

### Setup

For demonstration purposes, we will make use of the [Gisette](https://archive.ics.uci.edu/ml/datasets/Gisette) dataset.

To populate the ``data/`` folder with the necessary data, simply run*:

```shell
sh datasets.sh
```

You should end up with the following data files:

```shell
data/
    gisette_scale  # extracted from gisette_scale.bz2
    gisette_scale.t  # extracted from gisette_scale.t.bz2
```

To enable reproducibility, [Poetry](https://python-poetry.org/) has been used as a dependency manager.

```shell
python3 -m pip install poetry
```

and then:

```shell
python3 -m poetry install --no-dev
```

to install all required project dependencies in a virtual environment.

### Usage

Spawn a shell within the created virtual environment with:

```shell
python3 -m poetry shell
```

From within the shell, the following:

```shell
python cli.py --help
```

will guide you through the available options:

```shell
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  fit   Perform a simple training of the SMO-based classifier, given a C.
  tune  Perform a hyperparameter tuning of the SMO-based classifier.
```

So, for example:

```
python cli.py tune
```

will perform a hyperparameter tuning of the SMO-based SVM classifier.

### Citation

```bibtex
@techreport{platt1998sequential,
author = {Platt, John},
title = {Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines},
institution = {Microsoft},
year = {1998},
month = {April},
abstract = {This paper proposes a new algorithm for training support vector machines: Sequential Minimal Optimization, or SMO. Training a support vector machine requires the solution of a very large quadratic programming (QP) optimization problem. SMO breaks this large QP problem into a series of smallest possible QP problems. These small QP problems are solved analytically, which avoids using a time-consuming numerical QP optimization as an inner loop. The amount of memory required for SMO is linear in the training set size, which allows SMO to handle very large training sets. Because matrix computation is avoided, SMO scales somewhere between linear and quadratic in the training set size for various test problems, while the standard chunking SVM algorithm scales somewhere between linear and cubic in the training set size. SMO's computation time is dominated by SVM evaluation, hence SMO is fastest for linear SVMs and sparse data sets. On real-world sparse data sets, SMO can be more than 1000 times faster than the chunking algorithm.},
url = {https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/},
number = {MSR-TR-98-14},
}
```

---

*command ``sh datasets.sh`` will probably not work on Windows - make necessary alterations depending on your OS.
