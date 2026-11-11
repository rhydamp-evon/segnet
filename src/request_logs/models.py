from django.db import models
from rest_framework_api_key.models import APIKey

class RequestLog(models.Model):
    user_id = models.IntegerField(null=True)
    api_key_prefix = models.CharField(null=True, max_length=10)
    request_method = models.CharField(max_length=10)
    request_path = models.CharField(max_length=500)
    response_status = models.IntegerField()
    request_body = models.JSONField(null=True)
    remote_address = models.CharField(max_length=50)
    server_hostname = models.CharField(max_length=50)
    run_time = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True, blank=True)

    class Meta:
        verbose_name = 'Request Log'
        verbose_name_plural = 'Request Logs'
        db_table = 'tbl_request_logs'
        default_permissions = ()

    def __str__(self):
        return self.request_path
