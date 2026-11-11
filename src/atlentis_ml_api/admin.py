from django.contrib import admin
from django.contrib.admin.sites import AdminSite
from atlentis_ml_api.settings import APP_ENV
from bluesky.models import BlueSkyRequest, BlueSkyVariant, BlueSkyProcessed
from training.models import TrainingSession


class MyAdminSite(AdminSite):
    site_header = "Atlentis Blue Sky"
    site_title = "Atlentis Blue Sky"
    index_title = "Atlentis Blue Sky"
    index_template = "admin/admin_panel_landing_page.html"

    def index(self, request, extra_context=None):
        extra_context = extra_context or {}
        extra_context['total_bluesky_requests'] = BlueSkyRequest.objects.count()
        extra_context['total_bluesky_variants'] = BlueSkyVariant.objects.count()
        extra_context['total_bluesky_processed'] = BlueSkyProcessed.objects.count()
        extra_context['total_training_session'] = TrainingSession.objects.count()
        extra_context['recent_bluesky_requests'] = BlueSkyRequest.objects.all(
        ).order_by('-id')[:5]
        extra_context['recent_bluesky_variant_requests'] = BlueSkyVariant.objects.all(
        ).order_by('-id')[:5]
        extra_context['recent_bluesky_processed_requests'] = BlueSkyProcessed.objects.all(
        ).order_by('-id')[:5]
        extra_context['recent_failed_requests'] = BlueSkyProcessed.objects.filter(
            is_processed=False).order_by('-created_at')[:5]
        return super(MyAdminSite, self).index(request, extra_context)
