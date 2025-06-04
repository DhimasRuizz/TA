from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name=''),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('upload/', views.upload_file, name='upload'),
    path('uploaded_documents/', views.uploaded_documents, name='uploaded_documents'),
    path('topterms/', views.show_top_terms_view, name='top_terms'),
    path('edit_keywords/<int:theme_id>/', views.edit_keywords_view, name='edit_keywords'),
    path('delete-document/<int:document_id>/', views.delete_document, name='delete_document'),
]
