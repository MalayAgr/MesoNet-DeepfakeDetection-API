import os

import numpy as np
from django.conf import settings
from django.core.files.storage import default_storage
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .serializers import PredictionSerializer

IMG_WIDTH = 256


class Prediction:
    def __init__(self, img_url, true_label, pred_label, probability, plots=None):
        self.img_url = img_url
        self.true_label = true_label
        self.pred_label = pred_label
        self.probability = probability
        self.plots = [] if plots is None else plots


def get_data_generator(batch_size=64, shuffle=True):
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    return test_datagen.flow_from_directory(
        directory=settings.DATA_DIRECTORY,
        target_size=(IMG_WIDTH, IMG_WIDTH),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=shuffle,
    )


def select_img_batch(batch_size):
    data = get_data_generator(batch_size)
    batch_idx = np.random.randint(0, data.samples // data.batch_size)

    imgs, labels = data[batch_idx]

    start = batch_idx * data.batch_size

    end = start + data.batch_size if start >= 0 else None
    indices = data.index_array[start:end]
    filenames = [os.path.join(settings.DATA_ROOT, data.filenames[i]) for i in indices]

    class_mapping = {value: key for key, value in data.class_indices.items()}

    return imgs, filenames, labels, class_mapping


def get_dataset_size():
    return get_data_generator().samples


def clr_to_tabular(clr):
    columns = ["", "precision", "recall", "f1-score", "support"]

    rows = []
    for key, value in clr.items():
        if key == "accuracy":
            support = clr["weighted avg"]["support"]
            accuracy = round(clr["accuracy"], 2)
            rows.append(["accuracy", "", "", accuracy, support])
            continue
        row = []
        row.append(key)
        row.extend((round(x, 2) for x in value.values()))
        rows.append(row)

    return {"columns": columns, "rows": rows}


def get_predictions(model, num_imgs, conv_idx):
    imgs, filenames, labels, label_map = select_img_batch(num_imgs)

    probs, preds = model.predict(imgs)

    indices = range(num_imgs)
    iterator = zip(filenames, labels, probs, preds, indices)

    details = []
    for filename, label, prob, pred, idx in iterator:
        details.append(
            Prediction(
                img_url=default_storage.url(filename),
                true_label=label_map[int(label)],
                pred_label=label_map[pred],
                probability=round((1 - prob if pred == 0 else prob) * 100, 4),
            )
        )

    if conv_idx:
        plot_urls = model.visualize_conv_layers(imgs, conv_idx)

        for idx, plots in enumerate(plot_urls):
            details[idx].plots = plots

    return details
