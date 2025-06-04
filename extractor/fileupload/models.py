from django.db import models
import uuid

class ProcessedDocument(models.Model):
    title = models.CharField(max_length=255)
    cleaned_text = models.TextField()
    theme = models.CharField(max_length=100)
    confidence_score = models.FloatField()
    uploaded_at = models.DateTimeField(auto_now_add=True)
    unique_id = models.CharField(max_length=255, unique=True, default=uuid.uuid4().hex)
    uploaded_file = models.FileField(upload_to='uploads/', unique=True, null=True, blank=True)

    def __str__(self):
        return self.title

class Theme(models.Model):
    name = models.CharField(max_length=255)
    fixed_keywords = models.TextField()  # For manually entered keywords
    dynamic_keywords = models.TextField(blank=True, null=True)  # For auto-generated dynamic keywords

    def __str__(self):
        return self.name