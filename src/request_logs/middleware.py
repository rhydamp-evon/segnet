import socket
import time
import json

from request_logs.models import RequestLog


class RequestLogMiddleware():
    def __init__(self, get_response):
        self.get_response = get_response
        self.start_time = time.time()

    def process_request(self, request):
        if request.method in ['POST', 'PUT', 'PATCH']:
            request.req_body = request.body
        return request

    def __call__(self, request):
        request = self.process_request(request)
        response = self.get_response(request)
        self.process_response(request, response)
        return response

    def get_api_key_prefix(self, request):
        header = request.META.get('HTTP_AUTHORIZATION', None)
        if header:
            tokens = header.split()
            if len(tokens) != 2:
                return None
            token = tokens[1]
            prefix = token.split('.')
            prefix = prefix[0] if len(prefix) != 0 else None
            return prefix
        return None
        
    def process_response(self, request, response):
        request_path = request.get_full_path()
        if not request_path.startswith('/api'):
            return response
        request_data = None
        content_type = request.META.get('CONTENT_TYPE', 'application/json')
        request_type = request.META.get('HTTP_ACCEPT', content_type)
        if request.method in ['POST', 'PUT', 'PATCH']:
            # Multipart uploads do not contain JSON.  Empty or malformed JSON
            # should not make request logging turn an API validation error into
            # a server error.
            if content_type.split(';', 1)[0] == 'application/json' and request.req_body:
                try:
                    request_data = json.loads(request.req_body)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    request_data = None
        data = {
            "user_id": request.user.pk,
            "request_method": request.method,
            "request_path": request_path,
            "response_status": response.status_code,
            "remote_address": request.META['REMOTE_ADDR'],
            "server_hostname": socket.gethostname(),
            "request_body": request_data,
            "api_key_prefix": self.get_api_key_prefix(request),
            "run_time": time.time() - self.start_time
        }
        RequestLog.objects.create(**data)
        return response
