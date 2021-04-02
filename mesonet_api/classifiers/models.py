import os
import uuid

from django.db import models


class MLModel(models.Model):
    def get_model_filename(self, filename):
        _, ext = os.path.splitext(filename)
        filename = '_'.join(self.model_name.lower().split())
        return os.path.join(
            f'{self.root}', f'{filename}{ext}'
        )

    root = 'trained_models'

    help_texts = {
        'model_name': 'A user-friendly name for the model',
        'model_desc': 'A short description of the model that can help the user make a decision',
        'model_file': 'The HDF5 containing the model'
    }

    model_id = models.UUIDField(
        'Model ID', default=uuid.uuid4, editable=False, primary_key=True)
    model_name = models.CharField(
        'Model Name', max_length=30, help_text=help_texts['model_name'], unique=True)
    model_desc = models.CharField(
        'Model Description', max_length=50, help_text=help_texts['model_desc'])
    model_file = models.FileField(
        upload_to=get_model_filename, help_text=help_texts['model_file'])

    class Meta:
        verbose_name = 'ML Model'
        verbose_name_plural = 'ML Models'

    def __str__(self):
        return self.model_name
