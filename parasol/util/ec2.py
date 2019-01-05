import tensorflow as tf
import os
import fabric
from fabric.connection import Connection
import fabric.transfer as transfer
import time
import socket
import tempfile
from path import Path
import shutil
import parasol
import boto3
from contextlib import contextmanager

ec2 = boto3.client('ec2')
s3 = boto3.resource('s3')

completed_requests = set()

gfile = tf.gfile

PEM_FILE = os.path.expanduser("~/.aws/umbrellas.pem")

COMMAND = """
from parasol.experiment import from_json;
from_json('%s').run()
"""

GPU_COMMAND = """
from deepx import T
T.set_default_device(T.gpu())
from parasol.experiment import from_json;
with T.device(T.gpu()):
    from_json('%s').run()
"""

def run_remote(params_path, gpu=False, instance_type='m5.large', ami='ami-0e3a4bae76a75fd3b', spot_price=0.5):
    command = COMMAND % params_path
    if gpu:
        ami = 'ami-0631e8abbfc486a1d'
        instance_type = 'g3.4xlarge'
        spot_price = 0.5
        command = GPU_COMMAND % params_path
    instance = request_instance(instance_type, ami, spot_price, params_path)
    with create_parasol_zip() as parasol_zip, Connection(instance, user="ubuntu", connect_kwargs={
        "key_filename": PEM_FILE
    }) as conn:
        print("Running remote experiment...")
        conn.put(parasol_zip)
        conn.run("mkdir parasol; unzip -o parasol.zip -d parasol; rm parasol.zip", hide='stdout')
        conn.run("PIPENV_YES=1 pipenv run python setup.py develop", hide='stdout')
        conn.run("echo \"%s\" > run.py" % command, hide='stdout')
        conn.run("tmux new-session -d -s 'experiment' \"pipenv run python run.py; sudo poweroff\"")

@contextmanager
def create_parasol_zip():
    parasol_dir = Path(parasol.__file__).parent
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        shutil.copytree(parasol_dir, tmpdir / 'parasol')
        tmparchive = Path(tmpdir) / 'parasol'
        os.system('find {dir} | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf'.format(
            dir=tmparchive
        ))
        os.system('cd {dir}/parasol; zip -r ../parasol.zip . >/dev/null'.format(
            dir=tmparchive.parent
        ))
        yield tmparchive.parent / "parasol.zip"

def get_spot_status(request_id):
    while True:
        try:
            waiter = ec2.get_waiter('spot_instance_request_fulfilled')
            waiter.wait(SpotInstanceRequestIds=[request_id])
            response = ec2.describe_spot_instance_requests(
                SpotInstanceRequestIds=[request_id]
            )
            return response['SpotInstanceRequests'][0]['InstanceId']
        except KeyboardInterrupt:
            break
        except KeyboardInterrupt:
            print('Waiting again...')

def get_instance_url(instance_id):
    response = ec2.describe_instances(
        InstanceIds=[instance_id]
    )
    return response['Reservations'][0]['Instances'][0]['PublicIpAddress']

def wait_on_ssh(instance_ip):
    connected = False
    while not connected:
        s = socket.socket()
        s.settimeout(4)
        try:
            s.connect((instance_ip, 22))
            connected = True
        except Exception:
            print('Failed to connect...')
            time.sleep(2)
        finally:
            s.close()

def request_instance(instance_type, ami, spot_price, instance_name):
    print("Requesting instance:", instance_type)
    response = ec2.request_spot_instances(
        AvailabilityZoneGroup='us-west-2',
        LaunchSpecification=dict(
            SecurityGroups=['rllab-sg'],
            ImageId=ami,
            InstanceType=instance_type,
            KeyName='umbrellas',
        ),
        SpotPrice=str(spot_price)

    )
    if len(response['SpotInstanceRequests']) > 0:
        for request in response['SpotInstanceRequests']:
            request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
            if request_id not in completed_requests:
                completed_requests.add(request_id)
                break
    else:
        request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
    instance_id = get_spot_status(request_id)
    print('Spun up instance:', instance_id)
    print('Setting instance properties')
    ec2.modify_instance_attribute(
        InstanceId=instance_id,
        BlockDeviceMappings=[{
            'DeviceName': '/dev/sda1',
            'Ebs': {
                'DeleteOnTermination': True
            }
        }]
    )
    ec2.create_tags(
        Resources=[instance_id],
        Tags=[
            { 'Key': "Name", 'Value': instance_name }
        ]
    )
    url = get_instance_url(instance_id)
    print('Instance IP:', url)
    wait_on_ssh(url)
    return url
