# SOLAR

This is the open source implementation of the representation learning and reinforcement learning method detailed in the following paper:

[Marvin Zhang](http://marvinzhang.com/)\*, [Sharad Vikram](http://www.sharadvikram.com/)\*, Laura Smith, Pieter Abbeel, Matthew J. Johnson, Sergey Levine.  
[**SOLAR: Deep Structured Representations for Model-Based Reinforcement Learning.**](https://arxiv.org/abs/1808.09105)  
[International Conference on Machine Learning](https://icml.cc/) (ICML), 2019.  
[Project webpage](https://sites.google.com/view/icml19solar)

For more information on the method, please refer to the [paper](https://arxiv.org/abs/1808.09105), [blog post](https://bair.berkeley.edu/blog/2019/05/20/solar/) and [talk](https://youtu.be/pCbs80XWQaY). For questions about this codebase, please contact the lead authors.

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
