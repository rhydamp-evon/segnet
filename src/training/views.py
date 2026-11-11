import requests

from django.http import JsonResponse
from django.db.models import Q
from django.template.response import TemplateResponse
from rest_framework.generics import UpdateAPIView, GenericAPIView
from rest_framework.response import Response
from rest_framework.views import APIView
from atlentis_ml_api.settings import TRAINING_SV_START_URL, USE_S3
from training.serializers import TrainingLogSerializer, TrainingStatusSerializer, TrainingConfigSerializer, OKResponseSerializer
from training.models import TrainingSession, TrainingSessionStatus
from training.tasks import start_training_server, stop_training_server
from sentry_sdk import capture_exception
from rest_framework_api_key.permissions import HasAPIKey
from rest_framework.permissions import IsAdminUser
import boto3

if USE_S3:
    from atlentis_ml_api.settings import AWS_STORAGE_BUCKET_NAME, AWS_S3_CUSTOM_DOMAIN


class StartTraining(GenericAPIView):
    '''
        Start remote training server and start training process
    '''
    serializer_class = OKResponseSerializer
    permission_classes = [HasAPIKey | IsAdminUser]

    def get_queryset(self):
        if getattr(self, 'swagger_fake_view', False):
            # queryset just for schema generation metadata
            return None

    def get(self, request, pk):

        # Check if there is any training with active status.
        # If there is no active training, start new training
        # else, return 401
        if TrainingSession.objects.filter(Q(status=TrainingSessionStatus.ACTIVE) | Q(status=TrainingSessionStatus.AWAITING)).exists():
            return JsonResponse({"error": "Some training session is already active."}, status=400)

        # Change the current status to AWAITING
        training_session = TrainingSession.objects.get(id=pk)
        if not training_session.status == TrainingSessionStatus.COMPLETED:
            training_session.status = TrainingSessionStatus.AWAITING
            training_session.is_recent_training = True
            training_session.save()

        # Start the AWS INSTANCE
        resp = requests.get(TRAINING_SV_START_URL)

        if resp.status_code != 200 and resp.text != "null":
            # Update on sentry
            raise Exception('Starting training aws instance failed.')

        # Start the training process
        start_training_server.apply_async(countdown=150)

        return JsonResponse({"status": "ok"}, status=200)


class StopTraining(GenericAPIView):
    '''
        Stop remote training server and stop training process.
        Training can be stopped only if the status is awaiting or active.
    '''
    serializer_class = OKResponseSerializer
    permission_classes = [HasAPIKey | IsAdminUser]

    def get_queryset(self):
        if getattr(self, 'swagger_fake_view', False):
            # queryset just for schema generation metadata
            return None

    def get(self, request, pk):
        training_session = TrainingSession.objects.get(id=pk)
        if training_session.status == TrainingSessionStatus.ACTIVE or training_session.status == TrainingSessionStatus.AWAITING:
            training_session.status = TrainingSessionStatus.STOPPED
        training_session.is_recent_training = False
        training_session.save()
        stop_training_server.delay()
        return Response({"status": "ok"})


class UpdateTrainingLog(UpdateAPIView):
    '''
    Update the training results per epoch.
    '''
    serializer_class = TrainingLogSerializer
    queryset = TrainingSession.objects.all()


class UpdateTrainingStatus(UpdateAPIView):
    '''
    When the training is started, the default training status is awaiting.
    The status is changed to active when the training starts (before first epoch)
    and changed to completed after last epoch.
    '''
    serializer_class = TrainingStatusSerializer
    queryset = TrainingSession.objects.all()


class NewTrainingConfig(APIView):
    '''
    When the training server starts, it uses this api endpoint to get the recent training configurations.
    Thus the training configuration for session with status as awaiting is returned.
    Ideally there should only be any 1 training with status 'AWAITING" or "ACTIVE" 
    '''

    def get(self, request):
        config = TrainingSession.objects.filter(
            status=TrainingSessionStatus.AWAITING).order_by('-created_at')
        serializer = TrainingConfigSerializer(config, many=True)
        return Response(serializer.data)


class AccuracyLineChartJSONView(GenericAPIView):
    '''
    This endpoint returns the data required for the accuracy per epoch charts on admin panel to show the training progress.
    Additional parameters required by the graph such as line color, chart background color are hardcoded 
    and sent in the same response data so that the charts remains consistent.
    '''

    serializer_class = OKResponseSerializer
    permission_classes = [HasAPIKey | IsAdminUser]

    def get_chart_data(self, data):
        ''' Returns list of epochs for label '''
        training_data = data['train']
        val_data = data['val']
        epochs_list = []
        train_acc = []
        train_loss = []
        val_acc = []
        val_loss = []

        chart_range = len(training_data)
        if len(training_data) != len(val_data):

            if len(training_data) > len(val_data):
                chart_range = len(val_data)
            else:
                chart_range = len(training_data)

        for i in range(chart_range):
            epochs_list.append(training_data[i]['epoch'])
            train_acc.append(training_data[i]['accuracy'])
            val_acc.append(val_data[i]['accuracy'])
            train_loss.append(training_data[i]['loss'])
            val_loss.append(val_data[i]['loss'])

        # Build the json data required for chart.
        chart_data = {}
        train_line = {
            "data": train_acc,
            "backgroundColor": "rgba(49,172,95,0.88)",
            "borderColor": "rgba(49,172,95,0.28)",
            "pointBackgroundColor": "rgba(49,172,95,0.88)",
            "pointBorderColor": "#fff",
            "label": "Training",
            "name": "Train"
        }

        val_line = {
            "data": val_acc,
            "backgroundColor": "rgba(241,0,233,0.47)",
            "borderColor": "rgba(241,0,79,0.96)",
            "pointBackgroundColor": "rgba(241,0,233,0.47)",
            "pointBorderColor": "#fff",
            "label": "Validation",
            "name": "Val"
        }

        chart_data["options"] = {
            "scales": {
                "xAxes": [{
                    "scaleLabel": {
                        "display": True,
                        "labelString": 'epochs'
                    }
                }],
                "yAxes": [{
                    "scaleLabel": {
                        "display": True,
                        "labelString": 'Accuracy'
                    }
                }]
            }
        }

        chart_data['labels'] = epochs_list
        chart_data['datasets'] = [train_line, val_line]

        return chart_data

    def get(self, request, pk):
        epochs = {}
        training_session = TrainingSession.objects.get(id=pk)
        json_data = training_session.training_logs
        try:
            epochs = self.get_chart_data(json_data)
        except Exception as e:
            capture_exception(e)
        return Response(epochs, status=200)


class LossLineChartJSONView(GenericAPIView):
    '''
    This endpoint returns the data required for the loss per epoch charts on admin panel to show the training progress.
    Additional parameters required by the graph such as line color, chart background color are hardcoded 
    and sent in the same response data so that the charts remains consistent.
    '''
    serializer_class = OKResponseSerializer
    permission_classes = [HasAPIKey | IsAdminUser]

    def get_chart_data(self, data):
        ''' Returns list of epochs for label '''
        training_data = data.get('train', [])
        val_data = data.get('val', [])
        epochs_list = []
        train_acc = []
        train_loss = []
        val_acc = []
        val_loss = []

        chart_range = len(training_data)
        if len(training_data) != len(val_data):

            if len(training_data) > len(val_data):
                chart_range = len(val_data)
            else:
                chart_range = len(training_data)

        for i in range(chart_range):
            try:
                epochs_list.append(training_data[i]['epoch'])

                train_acc.append(training_data[i]['accuracy'])
                val_acc.append(val_data[i]['accuracy'])

                train_loss.append(training_data[i]['loss'])
                val_loss.append(val_data[i]['loss'])
            except:
                pass
        # Build the json data required for chart.
        chart_data = {}

        train_loss_line = {
            "data": train_loss,
            "backgroundColor": "rgba(49,172,95,0.88)",
            "borderColor": "rgba(49,172,95,0.28)",
            "pointBackgroundColor": "rgba(49,172,95,0.88)",
            "pointBorderColor": "#fff",
            "label": "Training",
            "name": "TrainLoss"
        }

        val_loss_line = {
            "data": val_loss,
            "backgroundColor": "rgba(241,0,233,0.47)",
            "borderColor": "rgba(241,0,79,0.96)",
            "pointBackgroundColor": "rgba(241,0,233,0.47)",
            "pointBorderColor": "#fff",
            "label": "Validation",
            "name": "ValLoss"
        }
        chart_data['labels'] = epochs_list
        chart_data['datasets'] = [train_loss_line, val_loss_line]
        chart_data["options"] = {
            "scales": {
                "xAxes": [{
                    "scaleLabel": {
                        "display": True,
                        "labelString": 'epochs'
                    }
                }],
                "yAxes": [{
                    "scaleLabel": {
                        "display": True,
                        "labelString": 'loss'
                    }
                }]
            }
        }
        return chart_data

    def get(self, request, pk):
        training_session = TrainingSession.objects.get(id=pk)
        json_data = training_session.training_logs
        epochs = self.get_chart_data(json_data)
        return Response(epochs, status=200)


def model_evaluation_gallery(request, pk):
    #
    s3 = boto3.resource('s3')
    training_session = TrainingSession.objects.get(id=pk)
    my_bucket = s3.Bucket(AWS_STORAGE_BUCKET_NAME)
    s3_dir_url = f"https://{AWS_S3_CUSTOM_DOMAIN}" + \
        f"/assets/model_eval/{training_session.name}/"

    images_list = []

    for object_summary in my_bucket.objects.filter(Prefix="assets/test-images/"):
        # display_image = {
        #     "original_image_url": "",
        #     "mask_url": "",
        #     "processed_image_url": "",
        # }
        display_image = {}
        if object_summary.key[-1] == "/":
            continue

        image_url = object_summary.key
        image_name = image_url.split("/")[-1]

        display_image['aws_image_url'] = f"https://{AWS_S3_CUSTOM_DOMAIN}/{image_url}"

        display_image['processed_image_url'] = s3_dir_url + image_name
        display_image['mask_url'] = s3_dir_url + "mask-"+image_name

        images_list.append(display_image)

    context = {"model_name": training_session.name,
               "original_images": images_list}
    return TemplateResponse(request, 'admin/model_evaluation.html', context)
