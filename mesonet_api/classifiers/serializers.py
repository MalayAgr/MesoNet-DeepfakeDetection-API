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
            "conv_layers",
        ]


class PredictionSerializer(serializers.Serializer):
    img_url = serializers.URLField()
    true_label = serializers.CharField(max_length=20)
    pred_label = serializers.CharField(max_length=20)
    probability = serializers.FloatField()
    plots = serializers.ListField(child=serializers.URLField())
