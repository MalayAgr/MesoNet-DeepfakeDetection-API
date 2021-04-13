import json

from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
from sklearn.metrics import classification_report

from .models import MLModel
from .utils import get_data_generator


def evaluate_model(model):
    loaded_model = model.get_loaded_model()
    data = get_data_generator(shuffle=False)

    metrics = loaded_model.evaluate(data, verbose=0)
    metrics = dict(zip(loaded_model.metrics_names, metrics))

    _, preds = model.predict(data)

    return round(metrics["accuracy"] * 100, 2), classification_report(
        data.classes, preds
    )


def get_conv_layer_details(model):
    loaded_model = model.get_loaded_model()
    conv_layers = [layer for layer in loaded_model.layers if "conv" in layer.name]
    return [[layer.filters, layer.kernel_size] for layer in conv_layers]


@receiver(post_save, sender=MLModel)
def populate_accuracy_and_clr(sender, instance, created, **kwargs):
    if created:
        instance.accuracy, instance.clr = evaluate_model(instance)
        instance.conv_layers = json.dumps(get_conv_layer_details(instance))
        instance.save(update_fields=["accuracy", "clr", "conv_layers"])
