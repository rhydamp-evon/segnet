import requests
from atlentis_ml_api.settings import SLACK_WEBHOOK
import json


class PushNotification:
    """ Send Push Notification directly to the slack group """
    headers = {'content-type': 'application/json'}

    def __init__(self):
        self.webhook = SLACK_WEBHOOK
        self.headers = {'content-type': 'application/json'}

    def model_evaluation_started(self):
        """ Started Model evaluation """

        data = {
            "text": "Started model evaluation."
        }
        try:
            response = requests.post(
                self.webhook, json.dumps(data), headers=self.headers)
        except Exception as E:
            print("Error while sending Push Notification. ", E)

    def model_evaluation_completed(self):
        """ Started Model evaluation """

        data = {
            "text": "Model evaluation completed."
        }
        try:
            response = requests.post(
                self.webhook, json.dumps(data), headers=self.headers)
        except Exception as E:
            print("Error while sending Push Notification. ", E)
