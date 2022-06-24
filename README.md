# smo-from-scratch

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Implementing Sequential Minimization Optimization algorithm from [John C. Platt's 1998 paper](https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/).

To populate the ``data/`` folder with the necessary data, run:

### Setup

```shell
sh datasets.sh
```

**note**: this will probably not work on Windows - make necessary alterations depending on your OS.

You should end up with - at least - the following data files:

```shell
data/
    gisette_scale  # extracted from gisette_scale.bz2
    gisette_scale.t  # extracted from gisette_scale.t.bz2
    gisette_train.txt  # symbolic link to gisette_scale
    gisette_test.txt  # symbolic link to gisette_scale
```

### Usage

```shell
python cli.py --help
```

will guide you through the available options.