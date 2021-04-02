import os
import uuid

import matplotlib.pyplot as plt
import numpy as np
from django.conf import settings
from django.db import models
from mpl_toolkits.axes_grid1 import ImageGrid
from tensorflow.keras import Model as KerasModel
from tensorflow.keras.models import load_model

from .storages import MLModelStorage


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
        upload_to=get_model_filename,
        storage=MLModelStorage(),
        help_text=help_texts['model_file'])

    class Meta:
        verbose_name = 'ML Model'
        verbose_name_plural = 'ML Models'

    def __str__(self):
        return self.model_name

    def get_loaded_model(self):
        filepath = os.path.join(
            f'{settings.MODEL_FILE_ROOT}', f'{self.model_file.name}'
        )
        return load_model(filepath)

    def predict(model, data, steps=None, threshold=0.5):
        preds = model.predict(data, steps=steps, verbose=1)
        return preds, np.where(preds >= threshold, 1, 0)

    def get_activation_model(self, conv_idx):
        ml_model = self.get_loaded_model()
        conv_layers = [
            layer for layer in ml_model.layers if 'conv' in layer.name]
        selected_layers = [
            layer for index, layer in enumerate(conv_layers)
            if index in conv_idx
        ]
        activation_model = KerasModel(
            inputs=ml_model.inputs,
            outputs=[layer.output for layer in selected_layers]
        )
        return activation_model

    def _visualize_conv_layers_single_img(self, activations, conv_idx):
        images_per_row = 4

        for activation, idx in zip(activations, conv_idx):
            num_filters = activation.shape[-1]

            imgs = [activation[:, :, i] for i in range(num_filters)]

            num_rows = num_filters // images_per_row

            fig = plt.figure()
            grid = ImageGrid(fig, 111, (num_rows, images_per_row))

            for ax, im in zip(grid, imgs):
                ax.imshow(im, cmap='viridis')

            plt.title(f'Convolutional Layer {idx + 1}')
            plt.show()

    def visualize_conv_layers(self, imgs, conv_idx):
        activation_model = self.get_activation_model(conv_idx)
        activations = activation_model.predict(imgs)

        num_imgs = imgs.shape[0]
        num_layers = len(conv_idx)

        for idx in range(num_imgs):
            img_activs = [activations[i][idx, :, :, :]
                          for i in range(num_layers)]
            self._visualize_conv_layers_single_img(
                activations=img_activs, conv_idx=conv_idx
            )
