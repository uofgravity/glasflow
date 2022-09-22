# Contributing to glasflow

## Installation

To install `glasflow` and contribute clone the repo and install the additional dependencies with:

```shell
pip install -e .[dev,nflows]
```

**Note:** Make sure the submodules are up-to-date, see the [git documentation on submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) for details.

**Note:** because of a long-standing bug in `setuptools` ([#230](https://github.com/pypa/setuptools/issues/230)), editable installs (`pip install -e`) are not supported by older versions of `setuptools`. Editable installs require `setuptools>=64.0.3`. You may also require an up-to-date version of `pip`.

## Format checking

We use [pre-commit](https://pre-commit.com/) to re-format code using `black` and check the quality of code suing `flake8` before committing.

This requires some setup:

```shell
pip install pre-commit # Should already be installed
cd glasflow
pre-commit install
```

Now we you run `$ git commit` `pre-commit` will run a series of checks. Some checks will automatically change the code and others will print warnings that you must address and re-commit.

## Testing glasflow

When contributing code to `glasflow` please ensure that you also contribute corresponding unit tests and integration tests where applicable. We test `glasflow` using `pytest` and strive to test all of the core functionality in `glasflow`. Tests should be contained with the `tests` directory and follow the naming convention `test_<name>.py`. We also welcome improvements to the existing tests and testing infrastructure.

The tests can be run from the root directory using

```console
pytest
```

Specific tests can be run using

```console
pytest tests/test_<name>.py
```

**Note:** the configuration for `pytest` is pulled from `pyproject.toml`

See the `pytest` [documentation](https://docs.pytest.org/) for further details on how to write tests using `pytest`.

### Testing with nflows

The continuous integration in glasflow tests with the environment variable `GLASFLOW_USE_NFLOWS` set to `True` and `False` independently. We recommend running the test suite with both values prior opening a pull request. For example:

```console
$ export GLASFLOW_USE_NFLOWS=false
$ pytest
...
$ export GLASFLOW_USE_NFLOWS=true
$ pytest
...
```
