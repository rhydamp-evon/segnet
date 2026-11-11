from rest_framework import serializers
from rest_framework.exceptions import ValidationError

from training.models import TrainingSession, TrainingSessionStatus


class OKResponseSerializer(serializers.Serializer):
    '''
    Serializer to return OK status
    '''
    status = serializers.CharField(default='ok')


class TrainingLogSerializer(serializers.Serializer):
    '''
    Serializer for request body for training/:id/logs API endpoint.
    '''
    epoch = serializers.IntegerField()
    accuracy = serializers.FloatField()
    loss = serializers.FloatField()
    learning_rate = serializers.FloatField()
    train = serializers.BooleanField()

    def update(self, instance, validated_data):
        logs = instance.training_logs
        if validated_data['train']:
            logs['train'].append(validated_data)
            instance.training_logs = logs
            instance.save(update_fields=['training_logs', 'updated_at'])
            return validated_data
        else:
            logs['val'].append(validated_data)
            instance.training_logs = logs
            instance.save(update_fields=['training_logs', 'updated_at'])
            return validated_data


class TrainingStatusSerializer(serializers.Serializer):
    '''
    Serializer for request body for training/:id/update API endpoint.
    '''
    status = serializers.IntegerField()

    def validate_status(self, value):
        if not value in (TrainingSessionStatus.ACTIVE.value, TrainingSessionStatus.COMPLETED.value):
            raise ValidationError(
                "Invalid value. Only Allowed to change status to start/complted.")
        return value

    def update(self, instance, validated_data):
        instance.status = validated_data['status']
        instance.save(update_fields=['status', 'updated_at'])
        return instance


class TrainingConfigSerializer(serializers.ModelSerializer):
    '''
    Serializer to serialize response data for new/config API endpoint.
    '''
    class Meta:
        model = TrainingSession
        fields = ['id', 'name', 'learning_rate', 'batch_size', 'num_of_epochs',
                  'load_model', 'scheduler_step_size', 'scheduler_gamma']


class DataSetChildSerializer(serializers.Serializer):
    '''
    Child serializer used in TrainingDataSerializer. 
    '''
    data = serializers.FloatField()
    backgroundColor = serializers.CharField()
    borderColor = serializers.CharField()
    pointBackgroundColor = serializers.CharField()
    pointBorderColor = serializers.CharField()
    label = serializers.CharField()
    name = serializers.CharField()


class TrainingDatasetSerializer(serializers.Serializer):
    '''
    Child dataset used in TrainingChartJSONSerializer
    '''
    datasets = serializers.ListField(DataSetChildSerializer())


class TrainingChartJSONSerializer(serializers.Serializer):
    '''
    Main serializer for training json
    '''
    options = serializers.JSONField()
    dataset = serializers.JSONField(TrainingDatasetSerializer())
    labels = serializers.ListField(serializers.IntegerField())
