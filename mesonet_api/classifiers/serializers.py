from rest_framework import serializers

from .models import MLModel


class MLModelSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = MLModel
        fields = [
            "model_id",
            "model_name",
            "model_desc",
            "loss_curve",
            "accuracy",
            "clr",
        ]
