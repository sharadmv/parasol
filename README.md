# Parasol

## Installation
We use [https://pipenv.readthedocs.io/en/latest/](Pipenv) for dependency management.
```bash
$ pipenv install --ignore-pipfile
```

## Important files

- BLDS: `parasol/prior/blds.py`
- VAE: `parasol/model/vae.py`
- LQRFLM: `parasol/control/lqrflm.py`

## Running experiments

- `python experiments/vae/reacher-image/solar.py`
- `python experiments/solar/reacher-image/solar.py`
