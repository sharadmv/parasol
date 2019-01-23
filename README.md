# Parasol

## Installation
We use [Pipenv](https://pipenv.readthedocs.io/en/latest/) for dependency management.
```bash
$ pipenv install --ignore-pipfile
```

## Important files

- BLDS: `parasol/prior/blds.py`
- VAE: `parasol/model/vae.py`
- LQRFLM: `parasol/control/lqrflm.py`

## Running experiments

- `python experiments/vae/reacher-image/solar.py`
