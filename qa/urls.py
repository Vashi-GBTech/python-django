# qa/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('qa/', views.qa_view, name='qa_view'),
]
