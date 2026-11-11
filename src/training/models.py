from time import time
from django.utils import timezone
from django.db import models
from django.utils.translation import gettext as _
from core.models import TimestampModel


def get_default_session_name():
    current_date = timezone.now().strftime("%d%m%Y-%H%M%S")
    return f'Session-{current_date}'


def get_default_json_value():
    return {
        'train': [],
        'val': [],
    }


class TrainingSessionStatus(models.IntegerChoices):
    '''
        Available options for training status.
    '''
    NotStarted = 0, _('Not Started')
    AWAITING = 1, _('Awaiting')
    ACTIVE = 2, _('Active')
    PAUSED = 3, _('Paused')
    STOPPED = 4, _('Stopped')
    COMPLETED = 5, _('Completed')


class TrainingSession(TimestampModel):
    name = models.CharField(
        max_length=200, default=get_default_session_name, unique=True, help_text="Name of training, model name will be <name>.pth")
    learning_rate = models.FloatField(
        default=0.001, help_text="Learning rate of the model.")
    batch_size = models.IntegerField(
        default=8, help_text="Batch size for training.")
    num_of_epochs = models.IntegerField(
        default=400, help_text="Num of epochs for training.")
    load_model = models.BooleanField(
        default=True, help_text="When true , use existing model.")
    scheduler_step_size = models.IntegerField(
        default=100, help_text="Scheduler step size after which learning rate changes.")
    scheduler_gamma = models.FloatField(default=0.1)
    max_acc_model_path = models.URLField(null=True, editable=False)
    current_model_path = models.URLField(null=True, editable=False)
    training_logs = models.JSONField(default=get_default_json_value)
    status = models.SmallIntegerField(
        choices=TrainingSessionStatus.choices,
        default=TrainingSessionStatus.NotStarted,
        help_text="Current status of training.")
    is_recent_training = models.BooleanField(default=True)

    class Meta:
        db_table = 'training_session'
        ordering = ('-updated_at', '-created_at')
        permissions = [
            ('deploy_model', 'Can Deploy Model'),
            ('start_training', 'Can Start Training'), 
            ('stop_training', 'Can Stop Training'), 
            ('evaluate_model', 'Can Evaluate Model'), 
            ('view_evaluation_results', 'Can View Evaluation results')
        ]

    def __str__(self) -> str:
        return self.name
