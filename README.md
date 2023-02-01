[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7108558.svg)](https://doi.org/10.5281/zenodo.7108558)
[![PyPI](https://img.shields.io/pypi/v/glasflow)](https://pypi.org/project/glasflow/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/glasflow.svg)](https://anaconda.org/conda-forge/glasflow)

# Glasflow

glasflow is a Python library containing a collection of [Normalizing flows](https://arxiv.org/abs/1912.02762) using [PyTorch](https://pytorch.org). It builds upon [nflows](https://github.com/bayesiains/nflows).

## Installation

glasflow is available to install via `pip`:

```shell
pip install glasflow
```

or via `conda`:

```shell
conda install glasflow -c conda-forge
```

## PyTorch

**Important:** `glasflow` supports using CUDA devices but it is not a requirement and in most uses cases it provides little to no benefit.

By default the version of PyTorch installed by `pip` or `conda` will not necessarily match the drivers on your system, to install a different version with the correct CUDA support see the PyTorch homepage for instructions: <https://pytorch.org/>.

## Usage

To define a RealNVP flow:

```python
from glasflow import RealNVP

# define RealNVP flow. Change hyperparameters as necessary.
flow = RealNVP(
    n_inputs=2,
    n_transforms=5,
    n_neurons=32,
    batch_norm_between_transforms=True
)
```

Please see [glasflow/examples](https://github.com/uofgravity/glasflow/tree/main/examples) for a typical training regime example.

## nflows

glasflow uses a fork of nflows which is included as submodule in glasflow and can used imported as follows:

```python
import glasflow.nflows as nflows
```

It contains various bugfixes which, as of writing this, are not included in a current release of `nflows`.

### Using standard nflows

There is also the option to use an independent install of nflows (if installed) by setting an environment variable.

```shell
export  GLASFLOW_USE_NFLOWS=True
```

After setting this variable `glasflow.nflows` will point to the version of nflows installed in the current python environment.

**Note:** this must be set prior to importing glasflow.

## Contributing

Pull requests are welcome. You can review the contribution guidelines [here](https://github.com/uofgravity/glasflow/blob/main/CONTRIBUTING.md). For major changes, please open an issue first to discuss what you would like to change.

## Citing

If you use glasflow in your work please cite [our DOI](https://doi.org/10.5281/zenodo.7108558). We also recommend you also cite nflows following the guidelines in the [nflows readme](https://github.com/uofgravity/nflows#citing-nflows).
