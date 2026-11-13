from django.db import transaction
from rest_framework import generics, status
from rest_framework.response import Response

from object_editing.models import ObjectAnalysis, ObjectEditJob
from object_editing.serializers import (
    ObjectAnalysisCreateSerializer,
    ObjectAnalysisDetailSerializer,
    ObjectEditJobCreateSerializer,
    ObjectEditJobDetailSerializer,
)
from object_editing.tasks import task_analyze_object_image, task_process_object_edit


class ObjectAnalysisCreateAPIView(generics.CreateAPIView):
    queryset = ObjectAnalysis.objects.all()
    serializer_class = ObjectAnalysisCreateSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        analysis = serializer.save()
        transaction.on_commit(
            lambda: task_analyze_object_image.delay(str(analysis.id))
        )
        response = ObjectAnalysisDetailSerializer(analysis)
        return Response(response.data, status=status.HTTP_202_ACCEPTED)


class ObjectAnalysisDetailAPIView(generics.RetrieveAPIView):
    queryset = ObjectAnalysis.objects.prefetch_related("regions__parent")
    serializer_class = ObjectAnalysisDetailSerializer


class ObjectEditJobCreateAPIView(generics.GenericAPIView):
    serializer_class = ObjectEditJobCreateSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        job = serializer.save()
        reused_existing = getattr(serializer, "reused_existing", False)
        if not reused_existing:
            transaction.on_commit(
                lambda: task_process_object_edit.delay(str(job.id))
            )
        response = ObjectEditJobDetailSerializer(job)
        response_status = (
            status.HTTP_200_OK if reused_existing else status.HTTP_202_ACCEPTED
        )
        return Response(response.data, status=response_status)


class ObjectEditJobDetailAPIView(generics.RetrieveAPIView):
    queryset = ObjectEditJob.objects.select_related("analysis")
    serializer_class = ObjectEditJobDetailSerializer
