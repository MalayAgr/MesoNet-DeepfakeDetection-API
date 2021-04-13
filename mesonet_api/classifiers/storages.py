from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.utils.deconstruct import deconstructible


@deconstructible
class MLModelStorage(FileSystemStorage):
    def __init__(self, **kwargs):
        kwargs.update({"location": settings.MODEL_FILE_ROOT})
        super().__init__(**kwargs)
