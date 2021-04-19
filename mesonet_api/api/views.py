from classifiers.models import MLModel
from classifiers.serializers import MLModelSerializer, PredictionSerializer
from classifiers.utils import get_dataset_size, get_predictions
from django.shortcuts import get_object_or_404
from rest_framework import generics, permissions, views
from rest_framework.response import Response


class DatasetSizeView(views.APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request, *args, **kwargs):
        response = {"size": get_dataset_size()}
        return Response(response)


class ListModelsView(generics.ListAPIView):
    queryset = MLModel.objects.all()
    serializer_class = MLModelSerializer
    permission_classes = [permissions.AllowAny]


class PredictionResultsView(generics.ListAPIView):
    permission_classes = [permissions.AllowAny]
    serializer_class = PredictionSerializer

    def get_queryset(self):
        model_pk = self.kwargs["model_pk"]
        model = get_object_or_404(MLModel, pk=model_pk)

        conv_idx = self.kwargs.get("conv_idx", [])
        if conv_idx:
            conv_idx = [int(idx) for idx in conv_idx]

        num_imgs = self.kwargs["num_imgs"]

        return get_predictions(model, num_imgs, conv_idx)
