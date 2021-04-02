
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from .views import ListModelsView

urlpatterns = [
    path('available-models/', ListModelsView.as_view()),
    path('api-auth/', include('rest_framework.urls'))
]

