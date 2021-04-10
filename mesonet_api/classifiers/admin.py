from django.contrib import admin
from .models import MLModel

# Register your models here.
@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    readonly_fields = ['accuracy', 'clr']
