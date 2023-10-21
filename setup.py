#!/usr/bin/env python
"""The setup script."""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

# with open("HISTORY.md") as history_file:
#     history = history_file.read()

# NOTE: Update pinned_reqs whenever install_requires or extras_require changes.
install_requires = [
    "numpy>=1.17.0",
    "plotly>=5.17.0",
    "dash>=2.14.0",
]

extras_require = {
    "dev": [
        "pip>=20.3",
        "pylint==2.17.5",
        "yapf==0.40.1",

        # Testing
        "pytest==7.0.1",
        "pytest-cov==3.0.0",
        "pytest-benchmark==3.4.1",
        "pytest-xdist==2.5.0",
        "fire>=0.5.0",

        # Documentation
        "mkdocs==1.5.2",
        "mkdocs-material==9.2.8",
        "mkdocstrings[python]==0.23.0",
        "mkdocs-gen-files==0.5.0",

        # Distribution
        # "bump2version==1.0.1",
        # "wheel==0.40.0",
        # "twine==4.0.2",
        # "check-wheel-contents==0.4.0",
    ],
}

setup(
    author="",
    author_email="",
    classifiers=[],
    description="",
    install_requires=install_requires,
    extras_require=extras_require,
    license="MIT license",
    # long_description=readme + "\n\n" + history,
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="dp",
    name="dp",
    packages=find_packages(include=["dp", "dp.*"]),
    python_requires=">=3.7.0",
    test_suite="tests",
    url="",
    version="",
    zip_safe=False,
)
