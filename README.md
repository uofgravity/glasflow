# Glasflow

glasflow is a Python library containing a collection of [Normalizing flows](https://arxiv.org/abs/1912.02762) using [PyTorch](https://pytorch.org). It is inherited from [nflows](https://github.com/bayesiains/nflows). 

## Installation

To install from github, you need to clone this repository.

```bash
git clone https://github.com/igr-ml/glasflow.git
```

Then run: 

```bash
python setup.py install
```

## PyTorch
By default the version of PyTroch will not necessarily match the drivers on your system, to install a different version with the correct CUDA support see the PyTorch homepage for instructions: https://pytorch.org/.

## Usage

To define a RealNVP flow:
```python
from glasflow.flows import RealNVP

# define RealNVP flow. Change hyperparameters as nessesary.
flow = RealNVP(n_inputs=2,
        n_transforms=5,
        n_neurons=32,
        batch_norm_between_transforms=True)
```

Please see [glasflow/examples](https://github.com/igr-ml/glasflow/tree/main/examples) for a typical training regime example. 

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

