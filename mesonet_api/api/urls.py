
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

from .views import ListModelsView, PredictionResultsView

urlpatterns = [
    path('available-models/', ListModelsView.as_view()),
    path(
        'predictions/<uuid:model_pk>/<int:num_imgs>/<str:conv_idx>/',
        PredictionResultsView.as_view()
    ),
    path('api-auth/', include('rest_framework.urls'))
]
