from django.urls import path
from training import views

urlpatterns = [
    path('<int:pk>/start/', views.StartTraining.as_view()),
    path('<int:pk>/stop/', views.StopTraining.as_view()),
    path('<int:pk>/logs/', views.UpdateTrainingLog.as_view(),
         name='update-training-log'),
    path('<int:pk>/status/', views.UpdateTrainingStatus.as_view(),
         name='update-training-status'),
    path('new/config/', views.NewTrainingConfig.as_view(),
         name="recent-training-config"),
    path('<int:pk>/accuracyChartJSON/', views.AccuracyLineChartJSONView.as_view(),
         name='accuracy_line_chart_json'),
    path('<int:pk>/lossChartJSON/', views.LossLineChartJSONView.as_view(),
         name='loss_line_chart_json'),
]
