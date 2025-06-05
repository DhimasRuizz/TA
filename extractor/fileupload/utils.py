import re
import PyPDF2
import docx
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nlp_id.lemmatizer import Lemmatizer
from indoNLP.preprocessing import replace_word_elongation
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ExtractedSections:
    title: str
    abstract: str
    background: str
    conclusion: str

@dataclass
class ClassificationResult:
    theme_id: str
    theme_name: str
    confidence_score: float
    matching_keywords: Dict[str, float]

class TextProcessor:
    SECTION_PATTERNS = {
        "title": r'(?im)^(?:[A-Z][A-Z\s]+\n)?([^\n]+?(?=\s*(?:ABSTRAK|ABSTRACT|SKRIPSI|PROGRAM STUDI)))',
        "abstract": r'(?im)(?:ABSTRAK|ABSTRACT)[\s.:-]*\s*([^\n].*?)\s*(?=(?:Kata Kunci|Keywords|Latar Belakang|BAB I|1\. PENDAHULUAN))',
        "background": r'(Latar Belakang[\s\S]+?)(?=(?:Perumusan Masalah|Rumusan Masalah))',
        "conclusion": r'(Kesimpulan[\s\S]+?)(?=(?:Saran|Daftar Pustaka))'
    }

    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stopwords = set(nltk.corpus.stopwords.words('indonesian'))
        self.lemmatizer = Lemmatizer()

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
            lowercase=True,
            token_pattern=r'(?u)\b\w+\b'
        )

    def extract_pdf_text(self, file_path: Path) -> str:
        """Extract text dari PDF file."""
        try:
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                return " ".join(page.extract_text() for page in reader.pages)
        except Exception as e:
            raise ValueError(f"Gagal Mengekstrak PDF: {e}")

    def extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            doc = docx.Document(file_path)
            return "\n".join(para.text for para in doc.paragraphs)
        except Exception as e:
            raise ValueError(f"Gagal Mengekstrak DOCX : {e}")

    def extract_sections(self, text: str) -> ExtractedSections:
        """Extract sections dari dokumen text."""
        results = {}
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        
        try:
            for section, pattern in self.SECTION_PATTERNS.items():
                match = re.search(pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
                content = match.group(1) if match else ""
                results[section] = re.sub(r'\s+', ' ', content).strip()
                
            return ExtractedSections(
                title=results.get("title", "Not found"),
                abstract=results.get("abstract", "Not found"),
                background=results.get("background", "Not found"),
                conclusion=results.get("conclusion", "Not found")
            )
        except Exception as e:
            raise ValueError(f"Gagal Mengextract sections: {str(e)}")

    def clean_text(self, text: str) -> str:
        """Clean dan preprocess text."""
        try:
            lemmatized = self.lemmatizer.lemmatize(text)
            processed = replace_word_elongation(lemmatized)
            tokens = nltk.word_tokenize(processed)
            tokens = [
                t.lower() for t in tokens 
                if t.isalpha() and t.lower() not in self.stopwords
            ]
            return " ".join(tokens)
        except Exception as e:
            raise ValueError(f"Text cleaning gagal: {e}")

    def classify_single_text(self, document: str, themes: Dict[str, Tuple[str, List[str]]], title: Optional[str] = None) -> ClassificationResult:
        try:
            # Clean dan preprocess
            cleaned_doc = self.clean_text(document)
            
            # Vectorize cleaned text
            X = self.vectorizer.fit_transform([cleaned_doc])
            feature_names = self.vectorizer.get_feature_names_out()
            X_dense = X.toarray()[0]

            scores = {}
            matching_keywords = defaultdict(dict)

            for theme_id, (theme_name, keywords) in themes.items():
                theme_score = 0
                keyword_matches = {}

                # Clean dan preprocess keywords
                clean_keywords = [self.clean_text(kw) for kw in keywords]

                for kw, original_kw in zip(clean_keywords, keywords):
                    # Mencari keyword dalam dokumen
                    keyword_score = 0
                    for i, term in enumerate(feature_names):
                        if kw == term:
                            keyword_score += X_dense[i] * 3.0
                        elif kw in term or term in kw:
                            similarity = len(set(kw.split()) & set(term.split())) / max(len(kw.split()), len(term.split()))
                            if similarity > 0.5:
                                keyword_score += X_dense[i] * similarity * 1.5
                    
                    if keyword_score > 0:
                        keyword_matches[original_kw] = keyword_score
                        theme_score += keyword_score

                # Jika ada keyword yang cocok dalam judul menambah skor
                if title:
                    clean_title = self.clean_text(title)
                    title_boost = sum(2.0 for kw in clean_keywords if kw in clean_title)
                    theme_score *= (1.0 + 0.3 * title_boost)

                scores[theme_id] = theme_score

            if not scores:
                raise ValueError("Tidak ditemukan tema yang cocok dengan dokumen.")

            # Mencari tema dengan skor tertinggi
            best_theme_id = max(scores, key=scores.get)
            best_score = scores[best_theme_id]
            
            # Menghitung skor normalisasi confidence score
            total_scores = sum(scores.values())
            normalized_score = best_score / total_scores if total_scores > 0 else 0
            normalized_score = max(0.0, min(1.0, normalized_score))

            return ClassificationResult(
                theme_id=best_theme_id,
                theme_name=themes[best_theme_id][0],
                confidence_score=normalized_score,
                matching_keywords=dict(matching_keywords.get(best_theme_id, {}))
            )

        except Exception as e:
            raise ValueError(f"Klasifikasi Gagal : {str(e)}")

    # def classify_single_text(self, document: str, themes: Dict[str, Tuple[str, List[str]]], title: Optional[str] = None) -> ClassificationResult:
    #     try:
    #         # Clean and preprocess document
    #         cleaned_doc = self.clean_text(document)
            
    #         # Vectorize cleaned text
    #         doc_vector = self.vectorizer.fit_transform([cleaned_doc])
    #         feature_names = self.vectorizer.get_feature_names_out()
            
    #         scores = {}
    #         matching_keywords = defaultdict(dict)

    #         for theme_id, (theme_name, keywords) in themes.items():
    #             theme_score = 0
    #             keyword_matches = {}

    #             # Clean and preprocess keywords
    #             clean_keywords = [self.clean_text(kw) for kw in keywords]
                
    #             for kw, original_kw in zip(clean_keywords, keywords):
    #                 # Vectorize keyword
    #                 kw_vector = self.vectorizer.transform([kw])
                    
    #                 # Calculate cosine similarity between document and keyword
    #                 similarity = cosine_similarity(doc_vector, kw_vector)[0][0]
                    
    #                 # Apply weight to exact matches
    #                 if kw in cleaned_doc:
    #                     similarity *= 1.5
                    
    #                 if similarity > 0:
    #                     keyword_matches[original_kw] = similarity
    #                     theme_score += similarity
                
    #             # Title boosting if available
    #             if title:
    #                 clean_title = self.clean_text(title)
    #                 title_vector = self.vectorizer.transform([clean_title])
                    
    #                 # Check keyword presence in title using cosine similarity
    #                 for kw, original_kw in zip(clean_keywords, keywords):
    #                     kw_vector = self.vectorizer.transform([kw])
    #                     title_similarity = cosine_similarity(title_vector, kw_vector)[0][0]
                        
    #                     if title_similarity > 0.5:
    #                         theme_score *= (1.0 + 0.3 * title_similarity)
                
    #             scores[theme_id] = theme_score
    #             matching_keywords[theme_id] = keyword_matches

    #         if not scores:
    #             raise ValueError("Tidak ditemukan tema yang cocok dengan dokumen.")

    #         # Find theme with highest score
    #         best_theme_id = max(scores, key=scores.get)
    #         best_score = scores[best_theme_id]
            
    #         # Normalize confidence score
    #         total_scores = sum(scores.values())
    #         normalized_score = best_score / total_scores if total_scores > 0 else 0
    #         normalized_score = max(0.0, min(1.0, normalized_score))

    #         return ClassificationResult(
    #             theme_id=best_theme_id,
    #             theme_name=themes[best_theme_id][0],
    #             confidence_score=normalized_score,
    #             matching_keywords=dict(matching_keywords.get(best_theme_id, {}))
    #         )

    #     except Exception as e:
    #         raise ValueError(f"Klasifikasi Gagal : {str(e)}")
        
    def get_similar_documents(self, input_text: str, other_docs: List[Tuple[str, str]], top_n: int = 5) -> List[Tuple[int, float, str]]:
        """
        Mencari Dokumen yang mirip dengan dokumen input berdasarkan TF-IDF similarity.
        """
        try:
            if not other_docs:
                return []
            
            # Klasifikasi dokumen yang input
            input_theme = self.classify_single_text(input_text, self._get_theme_dict()).theme_name
            
            # Filter dokumen lain berdasarkan tema yang sama
            same_theme_docs = [
                (text, theme) for text, theme in other_docs 
                if theme == input_theme
            ]
            
            # Jika tidak ada dokumen dengan tema yang sama, kembalikan daftar kosong
            if not same_theme_docs:
                return []
            
            # Pisahkan teks dan tema dari dokumen yang sama
            doc_texts, doc_themes = zip(*same_theme_docs)
            
            # Clean
            cleaned_input = self.clean_text(input_text)
            cleaned_docs = [self.clean_text(doc) for doc in doc_texts]
            
            # Calculate TF-IDF similarity
            corpus = [cleaned_input] + cleaned_docs
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            # Get top N documents
            top_indices = cosine_sim.argsort()[::-1][:top_n]
            
            # Return hasil dengan scores dan tema
            return [
                (int(i), 
                float(cosine_sim[i]), 
                doc_themes[i]
                ) for i in top_indices
            ]
        except Exception as e:
            raise ValueError(f"Similarity calculation failed: {e}")

    def _get_theme_dict(self) -> Dict[str, Tuple[str, List[str]]]:
        """Helper untuk mengambil dict tema dari database."""
        from .models import Theme
        themes = Theme.objects.all()
        return {
            str(theme.id): (
                theme.name,
                theme.get_fixed_keywords() + theme.get_dynamic_keywords()
            )
            for theme in themes
        }

    def calculate_top_terms(self, documents: List[str], theme_name: str) -> List[Tuple[str, float]]:
        """Calculate top TF-IDF terms."""
        try:
            if not documents:
                return []
            
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            feature_names = self.vectorizer.get_feature_names_out()
            avg_scores = tfidf_matrix.mean(axis=0).A1
            
            top_terms = sorted(
                zip(feature_names, avg_scores),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            return top_terms
        except Exception as e:
            print(f"Error calculating TF-IDF for theme {theme_name}: {e}")
            return []