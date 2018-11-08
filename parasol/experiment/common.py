import six
import json
from path import Path
from abc import ABCMeta, abstractmethod

import tensorflow.gfile as gfile

@six.add_metaclass(ABCMeta)
class Experiment(object):

    def __init__(self, experiment_name, out_dir='s3://parasol-experiments/'):
        self.experiment_name = experiment_name
        self.out_dir = Path(out_dir) / self.experiment_name

    def to_json(self):
        return json.dumps(self.to_dict())

    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def from_dict(self):
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
                fp.write(self.to_json())
            self.run_experiment(self.out_dir)

    @abstractmethod
    def run_experiment(self, out_dir):
        pass

    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, state):
        self.__dict__.update(self.from_dict(state).__dict__)

    def __eq__(self, other):
        return self.to_json() == other.to_json()
