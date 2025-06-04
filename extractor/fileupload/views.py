from django.shortcuts import render, get_object_or_404, redirect
from django.http import FileResponse, Http404
from django.core.files.storage import FileSystemStorage
from .utils import (
    extract_pdf_text, extract_docx_text, extract_sections,
    clean_text_pipeline, classify_single_text, get_similar_documents, calculate_top_tfidf_terms
)
from .forms import ThemeForm
from .models import ProcessedDocument, Theme
from django.db import models
from django.contrib import messages
from django.db.models import Count

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid

def generate_unique_id():
    while True:
        unique_id = uuid.uuid4().hex
        if not ProcessedDocument.objects.filter(unique_id=unique_id).exists():
            return unique_id

def upload_file(request):
    message = None
    cleaned_text = None
    result = None
    similar_docs = []  # Initialize similar_docs to avoid UnboundLocalError
    pdf_preview_url = None  # URL for the uploaded PDF preview
    success = False

    if request.method == 'POST' and request.FILES.get('uploaded_file'):
        uploaded_file = request.FILES['uploaded_file']
        fs = FileSystemStorage()

        unique_id = generate_unique_id()
        filename = unique_id + os.path.splitext(uploaded_file.name)[1]

        filename_saved = fs.save(filename, uploaded_file)
        file_path = fs.path(filename_saved)

        try:
            if filename.lower().endswith('.pdf'):
                raw_text = extract_pdf_text(file_path)
                message = "PDF berhasil diekstrak!"
                success = True
                pdf_preview_url = fs.url(filename_saved)
            elif filename.lower().endswith('.docx'):
                raw_text = extract_docx_text(file_path)
                message = "DOCX berhasil diekstrak!"
                success = True
            else:
                raw_text = ""
                message = "Format file tidak didukung. Silakan unggah file PDF atau DOCX."

            if raw_text:
                # Extract sections (title, abstract, etc.)
                sections = extract_sections(raw_text)
                title = sections.get("title", "").strip()
                if not title or title.lower() == "not found":
                    message = "Gagal mengekstrak judul dari dokumen."
                    fs.delete(filename)  # Delete the file if extraction failed
                    return render(request, 'upload.html', {
                        'message': message,
                        'success': success,
                        'pdf_preview_url': pdf_preview_url,  # Pass the preview URL for PDF
                        'docs': ProcessedDocument.objects.all()
                    })

                # Check for duplicate based on title
                if ProcessedDocument.objects.filter(title__iexact=title).exists():
                    message = f"Dokumen dengan judul '{title}' sudah ada."
                    success = False
                    fs.delete(filename)  # Delete the file if it's a duplicate
                    return render(request, 'upload.html', {
                        'message': message,
                        'pdf_preview_url': pdf_preview_url,  # Pass the preview URL for PDF
                        'docs': ProcessedDocument.objects.all()
                    })

                abstract = sections.get("abstract", "")
                background = sections.get("background", "")
                conclusion = sections.get("conclusion", "")
                cleaned_text = clean_text_pipeline(title, abstract, background, conclusion)

                # Get the themes and their keywords
                themes = Theme.objects.all()
                theme_dict = {}
                for theme in themes:
                    keywords = theme.fixed_keywords.split(",")  # Use fixed_keywords here
                    theme_dict[theme.id] = (theme.name, [kw.strip() for kw in keywords])

                # Classify the text into a theme using the classify_single_text function
                result = classify_single_text(cleaned_text, theme_dict, title)
                result['title'] = title

                # Save the document with unique ID and uploaded file path in the DB
                saved_doc = ProcessedDocument.objects.create(
                    title=title,
                    cleaned_text=cleaned_text,
                    theme=result['theme_name'],
                    confidence_score=result['confidence_score'],
                    unique_id=unique_id,
                    uploaded_file=filename_saved  # Store the uploaded file in the database
                )

                # Automatically update the keywords of the associated theme
                theme_to_update = Theme.objects.get(name=result['theme_name'])

                # Clean and update fixed keywords, removing duplicates and ensuring uniqueness
                new_keywords = theme_to_update.fixed_keywords.split(",")
                cleaned_keywords = [kw.strip() for kw in new_keywords if kw.strip()]  # Strip and remove empty entries
                cleaned_keywords = list(set(cleaned_keywords))  # Remove duplicates
                cleaned_keywords = cleaned_keywords[:15]  # Limit to 15 keywords

                theme_to_update.fixed_keywords = ', '.join(cleaned_keywords)  # Update the theme's fixed_keywords
                theme_to_update.save()

        except Exception as e:
            message = f"Error: {str(e)}"
            if 'filename' in locals():
                fs.delete(filename)

    return render(request, 'upload.html', {
        'message': message,
        'cleaned_text': cleaned_text,
        'result': result,
        'success': success,
        'pdf_preview_url': pdf_preview_url,  # Pass the preview URL for PDF
        'docs': ProcessedDocument.objects.all(),
        'similar_docs': similar_docs  # Pass similar_docs to the template
    })
    
def uploaded_documents(request):
    theme_filter = request.GET.get('theme', '')

    theme_counts = ProcessedDocument.objects.values('theme').annotate(count=Count('id'))
    theme_counts_dict = {item['theme']: item['count'] for item in theme_counts}
    total_documents = ProcessedDocument.objects.count()

    if theme_filter:
        documents = ProcessedDocument.objects.filter(theme=theme_filter)
    else:
        documents = ProcessedDocument.objects.all()

    themes = [
        "Computation & Artificial Intelligence",
        "Networking & Security",
        "Software Engineering & Mobile Computing",
        "Information System & Data Spatial"
    ]

    return render(request, 'uploaded_documents.html', {
        'documents': documents,
        'themes': themes,
        'selected_theme': theme_filter,
        'theme_counts': theme_counts_dict,
        'total_documents': total_documents
    })


from django.shortcuts import render
from .models import ProcessedDocument

def dashboard(request):
    themes = {
        0: "Computation & Artificial Intelligence",
        1: "Networking & Security",
        2: "Software Engineering & Mobile Computing",
        3: "Information System & Data Spatial"
    }

    theme_counts = {theme: ProcessedDocument.objects.filter(theme=theme).count() for theme in themes.values()}

    themes_count = [{'theme': theme, 'count': theme_counts.get(theme, 0)} for theme in themes.values()]

    total_documents = ProcessedDocument.objects.count()

    return render(request, 'dashboard.html', {'themes_count': themes_count, 'total_documents': total_documents})

def show_top_terms_view(request):

    themes = Theme.objects.all()
    documents = ProcessedDocument.objects.all()
    top_terms_data = calculate_top_tfidf_terms(documents)
    
    top_terms = []
    
    for theme in themes:
        terms = top_terms_data.get(theme.name, [])
        if terms:
            top_terms.append((theme, terms))
        else:
            top_terms.append((theme, "No top terms available"))
   
    return render(request, 'top_terms.html', {'top_terms': top_terms})


def edit_keywords_view(request, theme_id):
    theme = get_object_or_404(Theme, id=theme_id)

    # Get current fixed and dynamic keywords
    current_fixed_keywords = [keyword.strip() for keyword in theme.fixed_keywords.split(',')] if isinstance(theme.fixed_keywords, str) else theme.fixed_keywords
    current_dynamic_keywords = [keyword.strip() for keyword in theme.dynamic_keywords.split(',')] if isinstance(theme.dynamic_keywords, str) else theme.dynamic_keywords

    # Calculate top TF-IDF terms (dynamic keywords)
    theme_documents = ProcessedDocument.objects.filter(theme=theme.name)
    top_terms_data = calculate_top_tfidf_terms(theme_documents)  # Pass only theme-specific documents

    top_terms = []
    if theme.name in top_terms_data:
        top_terms = top_terms_data[theme.name][:5]  # Get top 5 terms specific to the theme

    auto_updated_keywords = [term[0] for term in top_terms]

    # Combine the fixed keywords with the auto-updated keywords
    fixed_keywords = current_fixed_keywords[:15]  # Keep only the first 15 from the user input

    # Clean the fixed keywords by removing extra spaces, ensuring no empty entries, and making them unique
    cleaned_keywords = [kw.strip() for kw in fixed_keywords if kw.strip()]
    cleaned_keywords = list(set(cleaned_keywords))  # Remove duplicates
    cleaned_keywords = cleaned_keywords[:15]  # Limit to 15 keywords

    dynamic_keywords = auto_updated_keywords  # Get only the dynamic top 5 keywords

    # Combine both fixed and dynamic keywords, then deduplicate across themes
    all_keywords = set(cleaned_keywords + dynamic_keywords)

    # Ensure no duplicates across all themes
    existing_keywords = set()
    for other_theme in Theme.objects.all():
        if other_theme.id != theme.id:  # Exclude the current theme
            existing_keywords.update(other_theme.fixed_keywords.split(','))
            existing_keywords.update(other_theme.dynamic_keywords.split(','))

    # Check for conflicts with existing keywords in other themes
    conflicting_keywords = all_keywords.intersection(existing_keywords)
    if conflicting_keywords:
        # Add a warning or error message about keyword conflicts
        messages.error(request, f"The following keywords are already used in another theme: {', '.join(conflicting_keywords)}")
        return redirect('edit_keywords', theme_id=theme.id)

    # Save the unique keywords to the current theme
    theme.fixed_keywords = ', '.join(cleaned_keywords)  # Store fixed keywords only
    theme.dynamic_keywords = ', '.join(dynamic_keywords)  # Store dynamic keywords only
    theme.save()

    # Add a success message for dynamic keywords update
    messages.success(request, 'Kata kunci untuk tema berhasil diperbarui dan tidak ada duplikat!')

    # Handling form submission
    if request.method == 'POST':
        form = ThemeForm(request.POST, instance=theme)
        if form.is_valid():
            # Get the raw fixed keywords from the form
            new_keywords_input = form.cleaned_data['fixed_keywords'].strip()
            if new_keywords_input:
                # Clean the fixed keywords, remove duplicates, and ensure uniqueness
                new_keywords = [kw.strip() for kw in new_keywords_input.split(',') if kw.strip()]
                fixed_keywords = list(set(new_keywords))  # Remove duplicates
                fixed_keywords = fixed_keywords[:15]  # Keep only the first 15 keywords

                # Save cleaned fixed keywords to the theme
                theme.fixed_keywords = ', '.join(fixed_keywords)  # Store fixed keywords only
                theme.save()

                # Add a success message for fixed keywords update
                messages.success(request, 'Kata kunci tetap berhasil diperbarui!')
            else:
                messages.error(request, 'Kata kunci tetap tidak boleh kosong!')

    else:
        form = ThemeForm(instance=theme)

    # Prepare data to pass to template
    theme_top_terms = [(term[0], term[1]) for term in top_terms]

    return render(request, 'edit_keywords.html', {
        'form': form,
        'theme': theme,
        'current_fixed_keywords': current_fixed_keywords,
        'current_dynamic_keywords': current_dynamic_keywords,
        'top_terms': theme_top_terms,
    })


def delete_document(request, document_id):
    if request.method == 'POST':
        document = get_object_or_404(ProcessedDocument, id=document_id)
        
        # Delete the physical file
        if document.uploaded_file and os.path.isfile(document.uploaded_file.path):
            try:
                os.remove(document.uploaded_file.path)
            except OSError as e:
                messages.error(request, f'Error deleting file: {e}')
                
        # Delete the database record
        document.delete()
        
        messages.success(request, 'File berhasil dihapus.')
        return redirect('uploaded_documents')