from celery import shared_task
from bluesky.services import BlueSkyProcessedService
from bluesky.models import BlueSkyProcessFailedLog, FailedProcessStatus
from sentry_sdk import capture_exception


@shared_task()
def task_process_image(data):
    try:
        service = BlueSkyProcessedService(data=data)
        return service.process_image()
    except Exception as e:
        print(e)
        capture_exception(e)
    return False


@shared_task()
def task_reprocess_failed_images(data):
    failed_log_id = data.pop('failed_log_id')
    success = task_process_image(data)
    if success:
        BlueSkyProcessFailedLog.objects.filter(id=failed_log_id) \
            .update(status=FailedProcessStatus.CLOSED.value)