# Dynamically Visualized (dynvis)

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
- Showing and highlighting the relevant lines of code used to compute the new
  entry in the animation.

## Installation

1. Clone the fork locally:

   ```bash
   # With SSH:
   git clone git@github.com:itsdawei/dynamically_programmed.git

   # Without SSH:
   git clone https://github.com/itsdawei/dynamically_programmed.git
   ```

1. Install the local copy into an environment. For instance, with
   [Conda](https://docs.conda.io/projects/miniconda/en/latest/), run the
   following commands:

   ```bash
   cd dynamically_programmed
   conda create --name=dp python=3.8 # 3.8 is the minimum version we support.
   conda activate dp
   conda install pip
   pip install .
   ```

## Contributors

dynvis is developed and maintained by

- Ramiro Deo-Campo Vuong
- Eric Han
- David H. Lee
- Aditya Prasad
- Tianyu Wang

## License

dynvis is released under the
[MIT License](https://github.com/itsdawei/dynamically_programmed/blob/main/LICENSE).
