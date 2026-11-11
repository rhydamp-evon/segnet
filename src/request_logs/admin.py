from django.contrib import admin
from request_logs.models import RequestLog

class RequestLogAdmin(admin.ModelAdmin):
    list_display = ('id', 'request_method', 'request_path',
                    'remote_address', 'run_time', 'timestamp')
    list_display_links = ('id', 'request_method', 'request_path')
    list_filter = ('request_method', 'timestamp')
    search_fields = ('request_method', )

admin.site.register(RequestLog, RequestLogAdmin)