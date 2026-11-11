from bluesky.services import BlueSkyProcessedService
from bluesky.models import BlueSkyProcessed

def test_push_to_api():
    service = BlueSkyProcessedService(data={})
    processed = BlueSkyProcessed.objects.last()
    service.push_results_to_api(processed)
    