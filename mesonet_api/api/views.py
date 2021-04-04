from classifiers.models import MLModel
from classifiers.serializers import MLModelSerializer
from classifiers.utils import get_dataset_size
from django.shortcuts import get_object_or_404, render
from rest_framework import generics, permissions, views
from rest_framework.response import Response


class DatasetSizeView(views.APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request, *args, **kwargs):
        response = {'size': get_dataset_size()}
        return Response(response)


class ListModelsView(generics.ListAPIView):
    queryset = MLModel.objects.all()
    serializer_class = MLModelSerializer
    permission_classes = [permissions.AllowAny]


class PredictionResultsView(views.APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request, *args, **kwargs):
        selected_model = get_object_or_404(MLModel, pk=kwargs['model_pk'])

        num_imgs = kwargs['num_imgs']
        conv_idx = [int(idx) for idx in kwargs['conv_idx']]

        response = selected_model.get_prediction_details(num_imgs, conv_idx)

        return Response(response)
