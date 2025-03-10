# Contributing

Contributions are welcome, and they are greatly appreciated. Every little bit
helps, and credit will always be given.

## Developing

Ready to contribute? Here's how to set up your environment for local development.

1. Clone the fork locally and change directory:
    ```bash
    # With SSH:
    git clone git@github.com:itsdawei/dpvis.git

    # Without SSH:
    git clone https://github.com/itsdawei/dpvis.git

    cd dpvis
    ```

1. There are a few options to managing virtual environment:
    - (Recommended) Install
      [Conda](https://docs.conda.io/projects/miniconda/en/latest/) and run:

      ```bash
      conda create -n dpvis python=3.11
      conda activate dpvis
      pip install -e .[dev]
      ```

    - Alternatively, install [Poetry](https://python-poetry.org/docs/) and run:

      ```bash
      poetry install
      ```

    - Although highly recommended, a virtual environment is not necessary as
      long as you have Python 3.9+ and a Python package manager such as
      [pip](https://pypi.org/project/pip/). In this case, you can install the
      library directly with:

      ```python
      pip install -e .[dev]
      ```

1. (Optional) Run knapsack demo to verify installation:

   ```bash
   poetry shell
   python demos/knapsack.py
   exit
   ```

1. Create a branch for local development:

   ```bash
   git checkout -b name-of-bugfix-or-feature
   ```

   Now make the appropriate changes locally.

   - Please follow the
     [Google Style Guide](https://google.github.io/styleguide/pyguide.html)
     (particularly when writing docstrings).
   - Make sure to auto-format the code using YAPF. We highly recommend
     installing an editor plugin that auto-formats on save, but YAPF also runs
     on the command line:

     ```bash
     yapf -i FILES
     ```

1. After making changes, check that the changes pass the tests:

   ```bash
   pytest tests/
   make test # ^ same as above
   ```

   Finally, to lint the code:

   ```bash
   pylint dp tests
   make lint # ^ same as above
   ```

   To get pytest and pylint, pip install them into the environment. However,
   they should already install with `pip install -e .[dev]`.

1. Add your change to the changelog for the current version in `HISTORY.md`.

1. Commit the changes and push the branch to GitHub:

   ```bash
   git add .
   git commit -m "Detailed description of changes."
   git push origin name-of-bugfix-or-feature
   ```

1. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before submitting a pull request, check that it meets these guidelines:

1. Style: Code should follow the
   [Google Style Guide](https://google.github.io/styleguide/pyguide.html) and be
   auto-formatted with [YAPF](https://github.com/google/yapf).
1. The pull request should include tests.
1. If the pull request adds functionality, corresponding docstrings and other
   documentation should be updated.
1. The pull request should work for Python 3.7 and higher. GitHub Actions will
   display test results at the bottom of the pull request page. Check there for
   test results.

## Instructions

### Running a Subset of Tests

To run a subset of tests, use `pytest` with the directory name, such as:

```bash
pytest tests/core/test1
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

### Referencing Papers

When referencing papers, refer to them as `Lastname YEAR`, e.g. `Smith 2004`.
Also, prefer to link to the paper's website, rather than just the PDF. This is
particularly relevant when linking to arXiv papers.
