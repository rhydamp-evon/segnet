from rest_framework.renderers import JSONRenderer

class CustomRenderer(JSONRenderer):
    def render(self, data, accepted_media_type=None, renderer_context=None):
        status_code = renderer_context['response'].status_code
        response = {
            "data": data,
            "status": True,
            "message": 'Success',
            "code": status_code
        }
        if not str(status_code).startswith('2'):
            response["status"] = False
            response["data"] = {}
            response["message"] = 'Failed'
            try:
                response["message"] = data["detail"]
            except KeyError:
                try:
                    response["message"] = data["error"]
                except KeyError:
                    response["data"] = data
        return super().render(response, accepted_media_type, renderer_context)