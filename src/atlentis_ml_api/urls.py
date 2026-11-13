from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.http.response import JsonResponse
from django.urls import path, include
from django.shortcuts import redirect
from atlentis_ml_api.settings import DEBUG

from drf_yasg.views import get_schema_view
from drf_yasg import openapi

from drf_spectacular.views import SpectacularAPIView, SpectacularRedocView, SpectacularSwaggerView

from training.views import model_evaluation_gallery

########### Swagger schema ###########
schema_view = get_schema_view(
    openapi.Info(
        title="Atlentis blue sky API",
        default_version='v1',
        description="Rest API documentation for Atlentis Blue Sky.",
    ),
    #    permission_classes=[permissions.IsAuthenticated],
    public=False
)


def index(request):
    return redirect('index')


def ping(request):
    return JsonResponse({"status": "ok"})


apiurlpatterns = [
    path('bluesky/', include('bluesky.urls')),
    path('object-editing/', include('object_editing.urls')),
    path('training/', include('training.urls')),
]

# if DEBUG:
swaggerurlpatterns = [
    # YOUR PATTERNS
    path('schema/', SpectacularAPIView.as_view(), name='schema'),
    # Optional UI:
    path('swagger/',
         SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('redoc/',
         SpectacularRedocView.as_view(url_name='schema'), name='redoc'),
]
apiurlpatterns = swaggerurlpatterns + apiurlpatterns


urlpatterns = [
    # path('', index),
    path('admin/training/<int:pk>/', model_evaluation_gallery,
         name="training-evaluation-results"),
    path('admin/', admin.site.urls),
    path('api/v1/', include(apiurlpatterns)),
    path('ping', ping)
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
