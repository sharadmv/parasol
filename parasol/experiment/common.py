import six
from path import Path
from abc import ABCMeta, abstractmethod

import tensorflow.gfile as gfile

from parasol.util import json
from parasol.experiment.remote import run_remote

@six.add_metaclass(ABCMeta)
class Experiment(object):

    def __init__(self, experiment_name, out_dir='out/'):
        self.experiment_name = experiment_name
        self.out_dir = Path(out_dir) / self.experiment_name

    @abstractmethod
    def to_dict(self):
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, params):
        pass

    def run(self, remote=False, gpu=False):
        if remote is False:
            if not gfile.Exists(self.out_dir):
                gfile.MakeDirs(self.out_dir)
            if not gfile.Exists(self.out_dir / "tb"):
                gfile.MakeDirs(self.out_dir / "tb")
            if not gfile.Exists(self.out_dir / "weights"):
                gfile.MakeDirs(self.out_dir / "weights")
            with gfile.GFile(self.out_dir / ("params.json".format(experiment_name=self.experiment_name)), 'w') as fp:
                json.dump(self, fp)
            self.run_experiment(self.out_dir)
        else:
            assert self.out_dir[:5] == "s3://", "Must be dumping to s3"
            with gfile.GFile(self.out_dir / "params.json", 'w') as fp:
                json.dump(self, fp)
            return run_remote(self.out_dir / "params.json")
        return self

    @abstractmethod
    def run_experiment(self, out_dir):
        pass

    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, state):
        self.__dict__.update(self.from_dict(state).__dict__)
