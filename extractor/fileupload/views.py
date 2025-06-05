from django.shortcuts import render, get_object_or_404, redirect
from django.http import FileResponse, Http404
from django.core.files.storage import FileSystemStorage
from .utils import TextProcessor
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
from typing import List, Dict
import os
import uuid

#DASHBOARD
def dashboard(request):
    themes = Theme.objects.all()
    theme_data = []
    
    for theme in themes:
        count = ProcessedDocument.objects.filter(theme=theme.name).count()
        theme_data.append({
            'theme': theme.name,
            'count': count
        })
    
    total_documents = ProcessedDocument.objects.count()
    
    return render(request, 'dashboard.html', {
        'themes_count': theme_data,
        'total_documents': total_documents
    })

#UPLOAD DOKUMEN
def generate_unique_id():
    while True:
        unique_id = uuid.uuid4().hex
        if not ProcessedDocument.objects.filter(unique_id=unique_id).exists():
            return unique_id

def upload_file(request):
    message = None
    cleaned_text = None
    result = None
    similar_results = []
    pdf_preview_url = None
    success = False
    processor = TextProcessor()
    title = None

    if request.method == 'POST' and request.FILES.get('uploaded_file'):
        uploaded_file = request.FILES['uploaded_file']
        fs = FileSystemStorage()
        
        unique_id = generate_unique_id()
        filename = unique_id + os.path.splitext(uploaded_file.name)[1]
        
        filename_saved = fs.save(filename, uploaded_file)
        file_path = fs.path(filename_saved)

        try:
            if filename.lower().endswith('.pdf'):
                raw_text = processor.extract_pdf_text(file_path)
                message = "PDF berhasil diekstrak!"
                success = True
                pdf_preview_url = fs.url(filename_saved)
            elif filename.lower().endswith('.docx'):
                raw_text = processor.extract_docx_text(file_path)
                message = "DOCX berhasil diekstrak!"
                success = True
            else:
                raw_text = ""
                message = "Format file tidak didukung. Silakan unggah file PDF atau DOCX."

            if raw_text:
                sections = processor.extract_sections(raw_text)
                title = sections.title
                
                if not title or title.lower() == "not found":
                    message = "Gagal mengekstrak judul dari dokumen."
                    fs.delete(filename)
                    return render(request, 'upload.html', {
                        'message': message,
                        'success': success,
                        'pdf_preview_url': pdf_preview_url,
                        'docs': ProcessedDocument.objects.all()
                    })

                if ProcessedDocument.objects.filter(title__iexact=title).exists():
                    message = f"Dokumen dengan judul '{title}' sudah ada."
                    success = False
                    fs.delete(filename)
                    return render(request, 'upload.html', {
                        'message': message,
                        'pdf_preview_url': pdf_preview_url,
                        'docs': ProcessedDocument.objects.all()
                    })

                # Clean dan process text
                combined_text = f"{title} {sections.abstract} {sections.background} {sections.conclusion}"
                cleaned_text = processor.clean_text(combined_text)

                # Ambil tema dan kata kunci
                themes = Theme.objects.all()
                theme_dict = {}
                for theme in themes:
                    fixed_keywords = theme.get_fixed_keywords()
                    dynamic_keywords = theme.get_dynamic_keywords()
                    all_keywords = fixed_keywords + dynamic_keywords
                    theme_dict[theme.id] = (theme.name, all_keywords)

                # Classify
                result = processor.classify_single_text(cleaned_text, theme_dict)

                # Save document
                saved_doc = ProcessedDocument.objects.create(
                    title=title,
                    cleaned_text=cleaned_text,
                    theme=result.theme_name,
                    confidence_score=result.confidence_score,
                    unique_id=unique_id,
                    uploaded_file=filename_saved
                )

                # Mengambil dokumen yang sudah ada untuk mencari kemiripan
                existing_docs = ProcessedDocument.objects.exclude(id=saved_doc.id)
                doc_data = [
                    (doc.cleaned_text, doc.theme) 
                    for doc in existing_docs
                ]

                similar_docs = processor.get_similar_documents(cleaned_text, doc_data)

                # Map 
                similar_results = [
                    {
                        'document': existing_docs[idx],
                        'score': f"{score:.2f}",
                        'theme': theme
                    }
                    for idx, score, theme in similar_docs
                ]

        except Exception as e:
            message = f"Error: {str(e)}"
            if 'filename' in locals():
                fs.delete(filename)
            success = False

    return render(request, 'upload.html', {
        'message': message,
        'cleaned_text': cleaned_text,
        'result': result,
        'success': success,
        'pdf_preview_url': pdf_preview_url,
        'docs': ProcessedDocument.objects.all(),
        'similar_docs': similar_results,
        'title': title
    })

#DATA DOKUMEN YANG SUDAH ADA
def uploaded_documents(request):
    theme_filter = request.GET.get('theme', '')
    
    if theme_filter:
        documents = ProcessedDocument.objects.filter(theme=theme_filter)
    else:
        documents = ProcessedDocument.objects.all()
    
    theme_counts = ProcessedDocument.objects.values('theme').annotate(count=Count('id'))
    theme_counts_dict = {item['theme']: item['count'] for item in theme_counts}
    
    return render(request, 'uploaded_documents.html', {
        'documents': documents,
        'selected_theme': theme_filter,
        'theme_counts': theme_counts_dict
    })

def delete_document(request, document_id):
    if request.method == 'POST':
        document = get_object_or_404(ProcessedDocument, id=document_id)
        
        if document.uploaded_file and os.path.isfile(document.uploaded_file.path):
            try:
                os.remove(document.uploaded_file.path)
            except OSError as e:
                messages.warning(request, f'Error menghapus file: {e}')
        
        document.delete()
        messages.success(request, 'Dokumen berhasil dihapus.')
        
    return redirect('uploaded_documents')

#TOP TERMS
def show_top_terms_view(request):
    processor = TextProcessor()
    themes = Theme.objects.all()
    top_terms_data = []

    if request.method == 'POST':
        for theme in themes:
            docs = ProcessedDocument.objects.filter(theme=theme.name)
            doc_texts = [doc.cleaned_text for doc in docs]
            terms = processor.calculate_top_terms(doc_texts, theme.name)
            
            # Mengambil fixed keywords yang sekarang
            fixed_keywords = set(theme.get_fixed_keywords())
            
            # Generate dynamic keywords
            dynamic_terms = [
                term for term, _ in terms 
                if term.lower() not in fixed_keywords
            ][:5]
            
            # Save dynamic keywords
            theme.set_dynamic_keywords(dynamic_terms)
            theme.save()
        
        messages.success(request, 'Kata kunci dinamis berhasil diperbarui!')
        return redirect('top_terms')

    for theme in themes:
        docs = ProcessedDocument.objects.filter(theme=theme.name)
        doc_texts = [doc.cleaned_text for doc in docs]
        terms = processor.calculate_top_terms(doc_texts, theme.name)
        
        # Ambil fixed and dynamic keywords sekarang
        fixed_keywords = theme.get_fixed_keywords()
        dynamic_keywords = theme.get_dynamic_keywords()
        
        top_terms_data.append({
            'theme': theme,
            'top_terms': terms,
            'fixed_keywords': fixed_keywords,
            'dynamic_keywords': dynamic_keywords
        })

    return render(request, 'top_terms.html', {
        'top_terms_data': top_terms_data
    })

#EDIT KATA KUNCI
def edit_keywords_view(request, theme_id):
    theme = get_object_or_404(Theme, id=theme_id)
    processor = TextProcessor()

    # Ambil dokumen yang terkait dengan tema ini
    theme_docs = ProcessedDocument.objects.filter(theme=theme.name)
    doc_texts = [doc.cleaned_text for doc in theme_docs]
    
    # Menghitung top terms
    # Jika tidak ada dokumen, top_terms akan kosong
    top_terms = processor.calculate_top_terms(doc_texts, theme.name) if doc_texts else []

    if request.method == 'POST':
        form = ThemeForm(request.POST, instance=theme)
        
        if form.is_valid():
            try:
                # Process fixed keywords
                fixed_keywords_input = form.cleaned_data['fixed_keywords']
                fixed_keywords = [k.strip() for k in fixed_keywords_input.split(',') if k.strip()]
                
                # Validate fixed keywords
                if len(fixed_keywords) != 15:
                    messages.error(request, "Harus ada tepat 15 kata kunci tetap")
                    return render(request, 'edit_keywords.html', {
                        'form': form,
                        'theme': theme,
                        'current_fixed_keywords': theme.get_fixed_keywords(),
                        'dynamic_keywords': top_terms,
                    })
                
                # Save fixed keywords
                theme.set_fixed_keywords(fixed_keywords)
                
                # Process dynamic keywords
                if top_terms:
                    # Filter fixed keywords dan dynamic keywords yang sudah ada
                    fixed_set = set(k.lower() for k in fixed_keywords)
                    dynamic_terms = []
                    
                    for term, _ in top_terms:
                        if (term.lower() not in fixed_set and 
                            len(dynamic_terms) < 5):
                            dynamic_terms.append(term)
                    
                    # Save dynamic keywords
                    theme.set_dynamic_keywords(dynamic_terms)
                
                # Save updated keywords
                theme.save()
                
                messages.success(request, 'Kata kunci berhasil diperbarui!')
                return redirect('top_terms')
                
            except ValueError as e:
                messages.error(request, str(e))
            except Exception as e:
                messages.error(request, f"Terjadi kesalahan: {str(e)}")
    else:
        form = ThemeForm(instance=theme)

    # Prepare dynamic keywords for display
    dynamic_keywords = []
    if top_terms:
        current_fixed = set(k.lower() for k in theme.get_fixed_keywords())
        dynamic_keywords = [
            (term, f"{score:.3f}")
            for term, score in top_terms
            if term.lower() not in current_fixed
        ][:5]

    return render(request, 'edit_keywords.html', {
        'form': form,
        'theme': theme,
        'current_fixed_keywords': theme.get_fixed_keywords(),
        'dynamic_keywords': dynamic_keywords,
        'has_documents': bool(doc_texts),
    })