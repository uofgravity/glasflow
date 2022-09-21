# Glasflow

glasflow is a Python library containing a collection of [Normalizing flows](https://arxiv.org/abs/1912.02762) using [PyTorch](https://pytorch.org). It builds upon [nflows](https://github.com/bayesiains/nflows).

## Installation

To install from GitHub:

```shell
$ pip install git+https://github.com/igr-ml/glasflow.git
```

## PyTorch

By default the version of PyTorch will not necessarily match the drivers on your system, to install a different version with the correct CUDA support see the PyTorch homepage for instructions: https://pytorch.org/.

## Usage

To define a RealNVP flow:

```python
from glasflow.flows import RealNVP

# define RealNVP flow. Change hyperparameters as nessesary.
flow = RealNVP(
    n_inputs=2,
    n_transforms=5,
    n_neurons=32,
    batch_norm_between_transforms=True
)
```

Please see [glasflow/examples](https://github.com/igr-ml/glasflow/tree/main/examples) for a typical training regime example.

## nflows

glasflow uses a fork of nflows which is included as submodule in glasflow and can used imported as follows:

```python
import glasflow.nflows as nflows
```

It contains various bugfixes which, as of writing this, are not included in current release of `nflows`.

### Using standard nflows

There is also the option to use an independent install of nflows (if installed) by setting an environment variable.

```shell
export  GLASFLOW_USE_NFLOWS=True
```

After setting this variable `glasflow.nflows` will point to the version of nflows installed in the current python environment.

**Note:** this must be set prior to importing glasflow.

## Contributing

Pull requests are welcome. You can review the contribution guidelines [here](https://github.com/igr-ml/glasflow/blob/main/CONTRIBUTING.md). For major changes, please open an issue first to discuss what you would like to change.

## Citing

If you use glasflow in your work we recommend you also cite nflows following the guidelines in the [nflows readme](https://github.com/igr-ml/nflows#citing-nflows).
