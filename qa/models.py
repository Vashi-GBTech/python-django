from django.contrib import admin
from django.db import models

class QAData(models.Model):
    context = models.TextField()
    question = models.TextField()
    answer = models.TextField()

admin.site.register(QAData)
