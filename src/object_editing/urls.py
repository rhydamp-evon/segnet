from django.urls import path

from object_editing import views


urlpatterns = [
    path(
        "analyses/",
        views.ObjectAnalysisCreateAPIView.as_view(),
        name="object-analysis-create",
    ),
    path(
        "analyses/<uuid:pk>/",
        views.ObjectAnalysisDetailAPIView.as_view(),
        name="object-analysis-detail",
    ),
    path(
        "jobs/",
        views.ObjectEditJobCreateAPIView.as_view(),
        name="object-edit-job-create",
    ),
    path(
        "jobs/<uuid:pk>/",
        views.ObjectEditJobDetailAPIView.as_view(),
        name="object-edit-job-detail",
    ),
]
