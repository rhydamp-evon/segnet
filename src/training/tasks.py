import sentry_sdk
from celery.decorators import task
import requests
from atlentis_ml_api.settings import TRAINING_SV_STOP_URL
from training.utils import TrainingServerHandler, restart_buddy_pipeline, move_training_pth_to_prod, download_pth_file
from sentry_sdk import capture_exception, capture_message
import os
from training.services import BlueSkyModelEvaluator


def is_supervisor_up(supervisor_status) -> bool:
    if supervisor_status['statecode'] == 1 and supervisor_status['statename'] == 'RUNNING':
        return True
    return False


@task(name="start-training-process")
def start_training_server():
    try:
        training_client = TrainingServerHandler()
        # training_client.get_supervisor_status()
        print("[+] Supervisor started, starting training.")
        if training_client.start_training():
            training_client = TrainingServerHandler()
    except Exception as e:
        capture_exception(e)


@task(name="stop-training-process")
def stop_training_server():
    try:
        # print("stopping training process")
        # training_client = TrainingServerHandler()
        # if training_client.stop_training():
        resp = requests.get(TRAINING_SV_STOP_URL)
        if resp.status_code == 200 and resp.text == "null":
            print("AWS Instance stopped.")
        else:
            raise Exception('Failed to stop training aws instance')
    except Exception as e:
        capture_exception(e)


@task(name="run-buddy-pipeline")
def run_buddy_pipeline(name):
    '''
    Hit the Buddy pipeline webhook, so that model file can be replaced on sv
    '''
    move_training_pth_to_prod(name)
    restart_buddy_pipeline()


@task(name="evaluate-model")
def evaluate_model(name):
    '''
    To evaluate model, we first download the training .pth to the api server
    once the file is copied to the api server, then we run the buddy pipeline using webhook
    After that the model is used for further processing, and results are saved in s3
    model evaluation is done on 30 images which are stores in /assets/test-images/
    '''

    if download_pth_file(name):
        checkpoint = name + ".pth"
        model_evaluation_subtask.apply_async(
            (checkpoint,), countdown=5*60)
        restart_buddy_pipeline()
    else:
        print("Failed to download pth file")
        capture_message("Failed to download new .pth file on api server.")


@task(name='model-evaluation-subtask')
def model_evaluation_subtask(checkpoint):
    training_evaluator = BlueSkyModelEvaluator(checkpoint)
    training_evaluator.evaluate()
