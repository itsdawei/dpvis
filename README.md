# dpvis

[![Documentation Status](https://readthedocs.org/projects/dpvis/badge/?version=latest)](https://dpvis.readthedocs.io/en/latest/?badge=latest)

The topic of dynamic programming (DP) is particularly challenging for learners
newly introduced to algorithm design. The library will serve as a learning aid
for studying DP. To this end, we design a widely accessible library that can be
used by students to visualize and interact with DP algorithms.

The ultimate goal of this library is not to implement visualization for any
finite collection of DP problems.
Instead, the library will have the capacity to visualize any DP algorithm,
provided that the algorithm is implemented correctly.

Our visualization library works with native Python implementations of DP.

The library has three major features:

- Capability to illustrate step-by-step executions of the DP as it fills out
  the dynamic programming array. This includes visualization of pre-computations
  and the backtracking process.
- Interactive self-testing feature in which the user will be quizzed on which
  cells will be used to compute the result, which cell will the result be
  stored, and what is the value of the result.

## Installation

**NOTE: This instruction should be updated after we release to PyPI.**

1. Clone the fork locally and change directory:

   ```bash
   # With SSH:
   git clone git@github.com:itsdawei/dpvis.git

   # Without SSH:
   git clone https://github.com/itsdawei/dpvis.git

   cd dpvis
   ```

1. (Optional) Create a virtual environment
   ([Conda](https://docs.conda.io/projects/miniconda/en/latest/)) and install
   poetry:

   ```bash
   conda create -n dpvis
   conda activate dpvis
   conda install poetry
   ```

1. Install library with poetry:

   ```bash
   poetry install
   ```

1. (Optional) Run knapsack demo to verify installation:

   ```bash
   poetry shell
   python demo/knapsack.py
   exit
   ```

## Documentation

The documentation is compiled with
[mkdocs-material](https://squidfunk.github.io/mkdocs-material/) and
[mkdocstrings](https://mkdocstrings.github.io/)

To serve the documentation locally, run

```bash
poetry install --with docs
poetry run make servedocs
```

## Contributors

dpvis is developed and maintained by

- Ramiro Deo-Campo Vuong
- Eric Han
- David H. Lee
- Aditya Prasad
- Tianyu Wang

## License

dpvis is released under the
[MIT License](https://github.com/itsdawei/dpvis/blob/main/LICENSE).
