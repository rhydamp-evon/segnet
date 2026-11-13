from django.test import RequestFactory, SimpleTestCase

from request_logs.middleware import RequestLogMiddleware


class RequestLogMiddlewareTests(SimpleTestCase):
    def test_long_invalid_api_key_prefix_is_safe_for_database_field(self):
        request = RequestFactory().post(
            "/api/v1/object-editing/jobs/",
            HTTP_AUTHORIZATION="Api-Key YOUR_API_KEY",
        )
        middleware = RequestLogMiddleware(lambda request: None)

        prefix = middleware.get_api_key_prefix(request)

        self.assertEqual(prefix, "YOUR_API_K")
