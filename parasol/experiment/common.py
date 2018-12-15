import traceback
import tempfile
import six
from path import Path
from abc import ABCMeta, abstractmethod

import tensorflow.gfile as gfile

from parasol.util import json, ec2, tee_out

@six.add_metaclass(ABCMeta)
class Experiment(object):

    def __init__(self, experiment_name, out_dir='out/'):
        self.experiment_name = experiment_name
        self.out_dir = Path(out_dir)

    @abstractmethod
    def to_dict(self):
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, params):
        pass

    def run(self, remote=False, gpu=False):
        out_dir = self.out_dir / self.experiment_name
        if remote is False:
            if not gfile.Exists(out_dir):
                gfile.MakeDirs(out_dir)
            if not gfile.Exists(out_dir / "tb"):
                gfile.MakeDirs(out_dir / "tb")
            if not gfile.Exists(out_dir / "weights"):
                gfile.MakeDirs(out_dir / "weights")
            with gfile.GFile(out_dir / "params.json", 'w') as fp:
                json.dump(self, fp)
            try:
                with tee_out(out_dir):
                    self.run_experiment(out_dir)
            except:
                traceback.print_exc()
                with gfile.GFile(out_dir / 'exception.log', 'w') as fp:
                    traceback.print_exc(file=fp)
        else:
            assert out_dir[:5] == "s3://", "Must be dumping to s3"
            return ec2.run_remote(out_dir / "params.json")
        return self

    @abstractmethod
    def run_experiment(self, out_dir):
        pass

    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, state):
        self.__dict__.update(self.from_dict(state).__dict__)
