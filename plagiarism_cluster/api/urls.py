from django.urls import path
from .views import (
    health, SimilarityView, BatchUploadView,
    AIDetectView, DatasetRunView, EvaluateView
)

urlpatterns = [
    path('health/',       health,                   name='health'),
    path('similarity/',   SimilarityView.as_view(),  name='similarity'),
    path('upload/',       BatchUploadView.as_view(), name='batch-upload'),
    path('ai-detect/',    AIDetectView.as_view(),    name='ai-detect'),
    path('dataset-run/',  DatasetRunView.as_view(),  name='dataset-run'),
    path('evaluate/',     EvaluateView.as_view(),    name='evaluate'),
]
