from .common import Controller

from .mpc import MPC

CONTROLLERS = [MPC]
CONTROL_MAP = {c.control_type: c for c in CONTROLLERS}

def from_config(model, control_params):
    control_params = control_params.copy()
    control_type = control_params.pop('control_type')
    return CONTROL_MAP[control_type](model, **control_params)
