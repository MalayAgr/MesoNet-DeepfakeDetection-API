from django.urls import include, path

from .views import ListModelsView, PredictionResultsView, DatasetSizeView

urlpatterns = [
    path("dataset-size/", DatasetSizeView.as_view()),
    path("available-models/", ListModelsView.as_view()),
    path(
        "predictions/<uuid:model_pk>/<int:num_imgs>/<str:conv_idx>/",
        PredictionResultsView.as_view(),
    ),
    path(
        "predictions/<uuid:model_pk>/<int:num_imgs>/", PredictionResultsView.as_view()
    ),
    path("api-auth/", include("rest_framework.urls")),
]
