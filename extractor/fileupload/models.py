from django.db import models
import uuid
import json

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
    fixed_keywords = models.TextField(blank=True, null=True)
    dynamic_keywords = models.TextField(blank=True, null=True)

    def get_fixed_keywords(self):
        return [k.strip() for k in self.fixed_keywords.split(',')] if self.fixed_keywords else []

    def get_dynamic_keywords(self):
        return [k.strip() for k in self.dynamic_keywords.split(',')] if self.dynamic_keywords else []

    def set_fixed_keywords(self, keywords):
        if len(keywords) != 15:
            raise ValueError("Must provide exactly 15 fixed keywords")

        cleaned_keywords = [k.strip() for k in keywords if k.strip()]
        self.fixed_keywords = ', '.join(cleaned_keywords)

    def set_dynamic_keywords(self, keywords):
        fixed_set = set(self.get_fixed_keywords())
        cleaned_keywords = [
            k.strip() for k in keywords 
            if k.strip() and k.strip().lower() not in fixed_set
        ]
        self.dynamic_keywords = ', '.join(cleaned_keywords[:5])