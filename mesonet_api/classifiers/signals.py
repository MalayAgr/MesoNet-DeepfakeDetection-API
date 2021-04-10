from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import MLModel
from .utils import get_data_generator


@receiver(post_save, sender=MLModel)
def create_profile(sender, instance, created, **kwrags):
    if created:
        model = instance.get_loaded_model()
        data = get_data_generator(shuffle=False)

        evaluation = model.evaluate(data, verbose=0)
        evaluation = dict(zip(model.metrics_names, evaluation))

        instance.accuracy = round(evaluation['accuracy'] * 100, 2)
        instance.save(update_fields=['accuracy'])
