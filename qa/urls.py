# qa/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.qa_view, name='qa_view'),
]
