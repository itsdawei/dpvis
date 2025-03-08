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

dpvis support Python 3.9 and above. The vast majority of users can install
dpvis by running:

```bash
pip install dpvis
```

You can test your installation by running one of our many demos.

```bash
python demos/knapsack.py
```

Then, open [http://127.0.0.1:8050/](http://127.0.0.1:8050/) with your favorite
browser (make sure that the prefix is "http" rather than "https").


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
