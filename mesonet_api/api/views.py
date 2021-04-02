from django.shortcuts import render
from rest_framework import generics
from rest_framework import permissions

from classifiers.models import MLModel
from classifiers.serializers import MLModelSerializer


class ListModelsView(generics.ListAPIView):
    queryset = MLModel.objects.all()
    serializer_class = MLModelSerializer
    permission_classes = [permissions.AllowAny]
