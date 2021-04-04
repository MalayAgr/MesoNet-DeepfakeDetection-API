import os

import numpy as np
from django.conf import settings
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_WIDTH = 256


def get_data_generator(batch_size=64):
    test_datagen = ImageDataGenerator(rescale=1./255)
    return test_datagen.flow_from_directory(
        directory=settings.DATA_DIRECTORY,
        target_size=(IMG_WIDTH, IMG_WIDTH),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )


def select_img_batch(batch_size):
    data = get_data_generator(batch_size)
    batch_idx = np.random.randint(0, data.samples // data.batch_size)

    imgs, labels = data[batch_idx]

    start = batch_idx * data.batch_size

    end = start + data.batch_size if start >= 0 else None
    indices = data.index_array[start:end]
    filenames = [
        os.path.join(settings.DATA_ROOT, data.filenames[i])
        for i in indices
    ]

    class_mapping = {value: key for key, value in data.class_indices.items()}

    return imgs, filenames, labels, class_mapping


def get_dataset_size():
    return get_data_generator().samples
