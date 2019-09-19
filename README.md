# Parasol

## Installation

We use [Pipenv](https://pipenv.readthedocs.io/en/latest/) for dependency management.
```bash
$ pipenv install
```

## Important files

- BLDS: `parasol/prior/blds.py` implements our global dynamics prior
- VAE: `parasol/model/vae.py` is our model with modular priors including unit Gaussian and BLDS
- LQRFLM: `parasol/control/lqrflm.py` implements our control method

## Running experiments

We provide a full example of the reacher experiment with the following files:

- `python experiments/vae/reacher-image/solar.py` trains our model and saves the weights
- `python experiments/solar/reacher-image/solar.py` loads the weights and runs SOLAR

Training the model is computationally expensive, slow, and
[does not always produce consistent results](https://github.com/sharadmv/parasol/issues/5).
Because of this, we also include a pretrained model in `data/vae/reacher-image/weights/model-final.pkl`.
These weights can be directly loaded and run with the second script. Please note that training a model
from scratch may lead to different policy performance, and multiple model training runs may be needed
in order to achieve good performance.

Note that this environment requires [OpenAI Gym](https://github.com/openai/gym) and related dependencies,
such as [MuJoCo](https://www.roboti.us/) and [mujoco-py](https://github.com/openai/mujoco-py).

