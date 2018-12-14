import base64
import deepx
import json
import pickle
import parasol

__all__ = ['dump', 'dumps', 'load', 'loads']

def dump(obj, fp):
    return json.dump(obj, fp, cls=ParasolJSONEncoder, indent=4, sort_keys=True)

def dumps(obj):
    return json.dumps(obj, cls=ParasolJSONEncoder, indent=4, sort_keys=True)

def load(fp):
    return json.load(fp, object_hook=decode_hook)

def loads(obj):
    return json.loads(obj, object_hook=decode_hook)

class ParasolJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, parasol.experiment.Experiment):
            return obj.to_dict()
        if isinstance(obj, deepx.core.Node):
            o = {
                '__bytes__': base64.b64encode(pickle.dumps(obj)).decode('ascii'),
                'readable': str(obj)
            }
            return o
        return super(ParasolJSONEncoder, self).default(obj)

def decode_hook(obj):
    if '__bytes__' in obj:
        return pickle.loads(base64.b64decode(obj['__bytes__'].encode('ascii')))
    return obj
