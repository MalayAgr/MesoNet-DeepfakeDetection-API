import os
import uuid
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.images import ImageFile
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
    plot_root = os.path.join(f'{settings.MEDIA_ROOT}', 'plots')

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
        if not hasattr(self, '_loaded_model'):
            filepath = os.path.join(
                f'{settings.MODEL_FILE_ROOT}', f'{self.model_file.name}'
            )
            self._loaded_model = load_model(filepath)
        return self._loaded_model

    def predict(self, data, steps=None, threshold=0.5):
        model = self.get_loaded_model()
        probs = model.predict(data, steps=steps)
        preds = np.where(probs >= threshold, 1, 0)

        probs = probs.reshape((probs.shape[0],))
        preds = preds.reshape((preds.shape[0],))
        return probs, preds

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

    @classmethod
    def save_plot(cls, f, filename_seed, idx):
        filepath = f'{filename_seed}_conv{idx}.png'
        filepath = os.path.join(cls.plot_root, filepath)
        filepath = default_storage.save(filepath, ImageFile(f))
        return default_storage.url(filepath)

    @classmethod
    def _visualize_conv_layers_single_img(cls, activations, conv_idx, filename_seed):
        images_per_row = 4

        urls = []
        for activation, idx in zip(activations, conv_idx):
            num_filters = activation.shape[-1]

            imgs = [activation[:, :, i] for i in range(num_filters)]

            num_rows = num_filters // images_per_row

            fig = plt.figure()
            grid = ImageGrid(fig, 111, (num_rows, images_per_row))

            for ax, im in zip(grid, imgs):
                ax.imshow(im, cmap='viridis')

            plt.title(f'Convolutional Layer {idx + 1}')

            f = BytesIO()
            plt.savefig(f, format='png')

            plot_url = cls.save_plot(f, filename_seed, idx)
            urls.append(plot_url)

            plt.clf()
        return urls

    def visualize_conv_layers(self, imgs, conv_idx):
        activation_model = self.get_activation_model(conv_idx)
        activations = activation_model.predict(imgs)

        num_imgs = imgs.shape[0]
        num_layers = len(conv_idx)

        urls = []
        for idx in range(num_imgs):
            img_activs = [activations[i][idx, :, :, :] for i in range(num_layers)]
            urls.append(self._visualize_conv_layers_single_img(
                activations=img_activs,
                conv_idx=conv_idx,
                filename_seed=f'plot_img{idx}'
            ))
        return urls

    def create_prediction_response(self, num_images, conv_idx):
        pass
