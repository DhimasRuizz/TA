from django import forms
from .models import Theme

class ThemeForm(forms.ModelForm):
    class Meta:
        model = Theme
        fields = ['fixed_keywords']  # Ensure you're using fixed_keywords, not keywords
        widgets = {
            'fixed_keywords': forms.Textarea(attrs={'placeholder': 'Enter your fixed keywords here'}),
        }