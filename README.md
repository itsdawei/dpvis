# dpvis

|                   Source                    |                       Docs                      |                    Paper                   |
| :-----------------------------------------: | :---------------------------------------------: | :------------------------------------------: |
| [GitHub](https://github.com/itsdawei/dpvis) | [dpvis.readthedocs.io](https://dpvis.readthedocs.io) | [arXiv](https://arxiv.org/abs/2411.07705) |

|                                                       PyPI                                           |                                                                                                      CI/CD                                              |                                                                   Docs Status                                                                 |
| :--------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------: |
| [![PyPI - Version](https://img.shields.io/pypi/v/dpvis?color=blue)](https://pypi.org/project/dpvis/) | [![tests](https://github.com/itsdawei/dpvis/actions/workflows/testing.yml/badge.svg)](https://github.com/itsdawei/dpvis/actions/workflows/testing.yml) | [![Documentation Status](https://readthedocs.org/projects/dpvis/badge/?version=latest)](https://dpvis.readthedocs.io/en/latest/?badge=latest) |

Dynamic programming (DP) is a fundamental and powerful algorithmic paradigm
taught in most undergraduate (and many graduate) algorithms classes.
DP problems are challenging for many computer science students because they
require identifying unique problem structures and a refined understanding of
recursion.
Dpvis is a Python library that helps students understand DP through
a frame-by-frame animation of dynamic programs.
Dpvis can easily generate animations of dynamic programs with as little as two
lines of modifications compared to a standard Python implementation.

Our visualization library works with native Python implementations of DP.

The library has two major features:

- Capability to illustrate step-by-step executions of the DP as it fills out
  the dynamic programming array. This includes visualization of pre-computations
  and the backtracking process.
- Interactive self-testing feature in which the user will be quizzed on which
  cells will be used to compute the result, which cell will the result be
  stored, and what is the value of the result.

## Easy Installation

Dpvis support Python 3.9 and above. The vast majority of users can install dpvis by running:

```bash
pip install dpvis
```

You can test your installation locally by running one of our many demos.

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
