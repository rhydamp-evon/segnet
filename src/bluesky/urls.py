from django.urls import path
from bluesky import views


urlpatterns = [
    path('sky-configs/', views.SkyConfigListView.as_view(),
         name='sky-config-list-api'),
    path('sky-configs/<int:pk>', views.SkyConfigDetailView.as_view(),
         name='sky-config-detail-api'),
    path('createRequest', views.CreateRequestAPIView.as_view(),
         name='create-request-api'),
    path('createVariant', views.CreateRequestVariantAPIView.as_view(),
         name='create-variant-api'),
    path('test-local-preview', views.LocalPreviewAPIView.as_view(),
         name='test-local-preview-api'),
    path('processVariant', views.ProcessVariantAPIView.as_view(),
         name='process-variant-api'),
    path('retryWebhook', views.RepushToWebhookAPIView.as_view(),
         name='retry-webhook-push'),
    path('procesFailedImages', views.ProcessFailedImagesAPIView().as_view(),
         name='process-failed-images')
]
