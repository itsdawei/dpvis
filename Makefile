# Convenient commands. Run `make help` for command info.
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@echo "\033[0;1mCommands\033[0m"
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[34;1m%-30s\033[0m %s\n", $$1, $$2}'

clean: clean-build clean-pyc clean-test clean-docker ## remove all build, test, coverage and Python artifacts
.PHONY: clean

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
.PHONY: clean-build

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
.PHONY: clean-pyc

clean-test: ## remove test and coverage artifacts
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
.PHONY: clean-test


# Clean all locally built images
clean-docker:
	@for demo in $(DEMOS); do docker rmi -f $(REPO)/$$demo || true; done
.PHONY: clean-docker

lint: ## check style with pylint
	pylint dp tests
.PHONY: lint

test: ## run tests with the default Python
	pytest tests
.PHONY: test

docs: ## generate HTML documentation, including API docs
	mkdocs build
	$(BROWSER) site/index.html
.PHONY: docs

servedocs: ## compile the docs watching for changes
	mkdocs serve
.PHONY: servedocs

# TODO Setup deployment to pip
# release-test: dist ## package and upload a release to TestPyPI
# 	twine upload --repository testpypi dist/*
# .PHONY: release-test

# release: dist ## package and upload a release
# 	twine upload dist/*
# .PHONY: release

# dist: clean ## builds source and wheel package
# 	python setup.py sdist
# 	python setup.py bdist_wheel
# 	ls -l dist
# 	check-wheel-contents dist/*.whl
