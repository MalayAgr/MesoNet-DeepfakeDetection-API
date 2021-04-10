from django.apps import AppConfig


class ClassifiersConfig(AppConfig):
    name = 'classifiers'

    def ready(self):
        import classifiers.signals
