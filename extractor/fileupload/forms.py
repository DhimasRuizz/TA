from django import forms
from .models import Theme

class ThemeForm(forms.ModelForm):
    class Meta:
        model = Theme
        fields = ['fixed_keywords']
        widgets = {
            'fixed_keywords': forms.Textarea(attrs={
                'class': 'border-gray-300 rounded-md shadow-sm w-full p-2',
                'rows': 5,
                'placeholder': 'Enter exactly 15 keywords separated by commas'
            })
        }

    def clean_fixed_keywords(self):
        keywords = [k.strip() for k in self.cleaned_data['fixed_keywords'].split(',') if k.strip()]

        if len(keywords) != 15:
            raise forms.ValidationError("Harus ada tepat 15 kata kunci")
        
        if any(len(k) < 2 for k in keywords):
            raise forms.ValidationError("Setiap kata kunci harus memiliki minimal 2 karakter")
        
        return ', '.join(keywords)