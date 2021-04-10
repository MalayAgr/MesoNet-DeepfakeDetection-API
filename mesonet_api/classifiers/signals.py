from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import MLModel



@receiver(post_save, sender=MLModel)
def create_profile(sender, instance, created, **kwargs):
    if created:
        instance.accuracy = instance.evaluate_model()
        instance.clr = instance.get_clr()

        print(instance.clr)

        instance.save(update_fields=['accuracy', 'clr'])
