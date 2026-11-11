import os
import time
import xmlrpc.client
import boto3
from django.utils import timezone
import requests
from atlentis_ml_api.settings import TRAINING_SUPERVISOR_USERNAME, TRAINING_SUPERVISOR_PASSWORD, \
    TRAINING_SERVER_IP, TRAINING_SERVER_PORT, USE_S3, BUDDY_WEBHOOK, BASE_DIR
from sentry_sdk import capture_message, capture_exception

if USE_S3:
    from atlentis_ml_api.settings import AWS_STORAGE_BUCKET_NAME


class TrainingServerHandler:
    '''
        Control the training server with supervisor xmlrpc api
        http://supervisord.org/api.html
    '''

    def __init__(self, server_ip=TRAINING_SERVER_IP, port=TRAINING_SERVER_PORT, username=TRAINING_SUPERVISOR_USERNAME, password=TRAINING_SUPERVISOR_PASSWORD):

        self.connection_string = f"http://{username}:{password}@{server_ip}:{port}/RPC2"
        try:
            self.supervisor_client = xmlrpc.client.ServerProxy(
                self.connection_string).supervisor
        except:
            pass

    def get_supervisor_status(self) -> dict:
        '''
        returns statecode and state name of remote supervisor instance
        ##################################################################
        statecode	statename	Description
            2	    FATAL	    Supervisor has experienced a serious error.
            1	    RUNNING	    Supervisor is working normally.
            0	    RESTARTING	Supervisor is in the process of restarting.
            -1	    SHUTDOWN	Supervisor is in the process of shutting down.
        ###################################################################
        '''
        return self.supervisor_client.getState()

    def start_training(self, name="start-training") -> bool:
        '''
        starts the training process on the training server 
        Returns True if started properly else false
        '''
        return self.supervisor_client.startProcess(name)

    def stop_training(self, name="start-training") -> bool:
        '''
        starts the training process on the training server 
        Returns True if started properly else false
        '''
        return self.supervisor_client.stopProcess(name)

    def is_training_active(self, name="start-training") -> bool:
        '''
        Returns if the training is active or not
        '''
        _process_info = self.supervisor_client.getProcessInfo(name)
        state_code = _process_info['state']
        state_name = _process_info['statename']
        if state_code == 20 and state_name == 'RUNNING':
            return True
        return False

    def latest_output(self, name="start-training") -> str:
        '''
            Returns the latest 500 bytes the log file
        '''
        log_text = self.supervisor_client.tailProcessStdoutLog(name, 0, 500)[0]
        return log_text

    def latest_error(self, name="start-training") -> str:
        '''
            Returns the latest 500 bytes the log file
        '''
        log_text = self.supervisor_client.tailProcessStderrLog(name, 0, 500)[0]
        return log_text


# client = TrainingServerHandler(server_ip, port, username, password)
# try:
#     client.start_training()
# except xmlrpc.client.Fault:
#     print("Training server is already running")

# print(client.get_supervisor_status())
# print(client.is_training_active())
# print(client.latest_output_error())
# print(client.stop_training())
# print(client.get_supervisor_status())

def move_training_pth_to_prod(training_session_name):
    '''
        When the deploy to production button is hit,
        the training .pth file is moved to history.
    '''
    s3_resource = boto3.resource('s3')

    try:
        # Copy current production file to history
        copy_from_path = "checkpoints/segnet/production/final_segmentation.pth"
        current_date = timezone.now().strftime("%d%m%Y-%H%M%S")
        copy_to_path = f"checkpoints/segnet/history/model-{current_date}.pth"
        s3_resource.Object(AWS_STORAGE_BUCKET_NAME, copy_to_path).copy_from(
            CopySource=copy_from_path)

        # copy training file to production
        copy_from_path = "checkpoints/segnet/training/{training_session_name}.pth"
        copy_to_path = "checkpoints/segnet/production/final_segmentation.pth"
        s3_resource.Object(AWS_STORAGE_BUCKET_NAME, copy_to_path).copy_from(
            CopySource=copy_from_path)

        # Delete the .pth file from training
        s3_resource.Object(AWS_STORAGE_BUCKET_NAME, copy_from_path).delete()

        capture_message("[+] Moving model to production Successful.")

    except Exception as e:
        capture_message("[!] Moving model to production failed.")
        capture_exception(e)


def restart_buddy_pipeline():
    try:
        response = requests.get(BUDDY_WEBHOOK)
        if response.status_code == 200:
            capture_message("[+] Buddy pipeline url run successfully.")
    except Exception as e:
        capture_exception(e)


def download_pth_file(filename) -> bool:
    s3 = boto3.resource('s3')
    filename = filename + ".pth"
    checkpoint_s3_key = f"checkpoints/segnet/training/{filename}"
    checkpoint_local_key = os.path.join(BASE_DIR, 'checkpoints', filename)
    try:
        s3.meta.client.download_file(
            AWS_STORAGE_BUCKET_NAME, checkpoint_s3_key, checkpoint_local_key)
        capture_message(
            f"successfully copied model file {filename} to api sv.")
        return True
    except Exception as e:
        capture_exception(e)
        return False
