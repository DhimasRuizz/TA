from django.contrib import admin
from .models import ProcessedDocument, Theme

admin.site.register(ProcessedDocument)
admin.site.register(Theme)