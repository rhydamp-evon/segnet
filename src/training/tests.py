from django.urls import reverse
from rest_framework.test import APITestCase
from rest_framework import status

from training.models import TrainingSession, TrainingSessionStatus

def create_training_session():
    session = TrainingSession.objects.create()
    return session


class APITestCase(APITestCase):

    update_training_log = 'update-training-log'
    update_training_status = 'update-training-status'

    def setUp(self) -> None:
        self.training_session = create_training_session()

    def test_update_training_log_api(self) -> None:
        url = reverse(self.update_training_log, kwargs={'pk': self.training_session.pk})
        data = {
            'epoch': 1,
            'accuracy': 10.0,
            'loss': 0.002,
            'learning_rate': 0.001
        }
        response = self.client.put(url, data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_update_start_training_status_api(self) -> None:
        url = reverse(self.update_training_status, kwargs={'pk': self.training_session.pk})
        data = {
            'status': TrainingSessionStatus.ACTIVE.value
        }
        response = self.client.put(url, data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], data['status'])

    def test_update_completed_training_status_api(self) -> None:
        url = reverse(self.update_training_status, kwargs={'pk': self.training_session.pk})
        data = {
            'status': TrainingSessionStatus.COMPLETED.value
        }
        response = self.client.put(url, data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], data['status'])

    def test_invalid_update_training_status_api(self) -> None:
        url = reverse(self.update_training_status, kwargs={'pk': self.training_session.pk})
        data = {
            'status': 1
        }
        response = self.client.put(url, data)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
