import os
import fabric
from fabric.connection import Connection
import fabric.transfer as transfer
import tensorflow as tf
from parasol.util import ec2

gfile = tf.gfile

PEM_FILE = os.path.expanduser("~/.aws/umbrellas.pem")

COMMAND = "from parasol.experiment import from_file; from_file(\\\"%s\\\").run()"

def run_remote(params_path, gpu=False):
    instance = '34.221.195.138' #ec2.request_instance('m5.4xlarge', 'ami-0a3085c8d39a07b81', 0.3, params_path)
    with ec2.create_parasol_zip() as parasol_zip, Connection(instance, user="ubuntu", connect_kwargs={
        "key_filename": PEM_FILE
    }) as conn:
        conn.put(parasol_zip)
        conn.run("mkdir parasol; unzip -o parasol.zip -d parasol; rm parasol.zip", hide='stdout')
        conn.run("PIPENV_YES=1 pipenv run python setup.py develop")
        command = COMMAND % params_path
        conn.run("tmux new-session -d -s 'experiment' \"pipenv run python -c '%s'; python\"" % command)
