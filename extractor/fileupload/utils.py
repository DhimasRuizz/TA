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
        """Extract text from PDF file."""
        try:
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                return " ".join(page.extract_text() for page in reader.pages)
        except Exception as e:
            raise ValueError(f"Error extracting PDF text: {e}")

    def extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            doc = docx.Document(file_path)
            return "\n".join(para.text for para in doc.paragraphs)
        except Exception as e:
            raise ValueError(f"Error reading DOCX file: {e}")

    def extract_sections(self, text: str) -> ExtractedSections:
        """Extract sections from document text."""
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
            raise ValueError(f"Failed to extract sections: {str(e)}")

    def clean_text(self, text: str) -> str:
        """Clean and preprocess text for Indonesian language."""
        try:
            # Lemmatize using Indonesian lemmatizer
            lemmatized = self.lemmatizer.lemmatize(text)
            # Replace word elongation (e.g., "selamattttt" -> "selamat")
            processed = replace_word_elongation(lemmatized)
            # Tokenize
            tokens = nltk.word_tokenize(processed)
            # Remove stopwords and non-alphabetic tokens
            tokens = [
                t.lower() for t in tokens 
                if t.isalpha() and t.lower() not in self.stopwords
            ]
            return " ".join(tokens)
        except Exception as e:
            raise ValueError(f"Text cleaning failed: {e}")

    def classify_single_text(self, document: str, themes: Dict[str, Tuple[str, List[str]]], title: Optional[str] = None) -> ClassificationResult:
        try:
            # Clean and preprocess the document
            cleaned_doc = self.clean_text(document)
            
            # Vectorize the cleaned document
            X = self.vectorizer.fit_transform([cleaned_doc])
            feature_names = self.vectorizer.get_feature_names_out()
            X_dense = X.toarray()[0]

            scores = {}
            matching_keywords = defaultdict(dict)

            for theme_id, (theme_name, keywords) in themes.items():
                theme_score = 0
                keyword_matches = {}

                # Clean and preprocess keywords
                clean_keywords = [self.clean_text(kw) for kw in keywords]

                for kw, original_kw in zip(clean_keywords, keywords):
                    # Find matches in document terms
                    keyword_score = 0
                    for i, term in enumerate(feature_names):
                        # Exact match with higher weight
                        if kw == term:
                            keyword_score += X_dense[i] * 3.0
                        # Partial match with lower weight
                        elif kw in term or term in kw:
                            similarity = len(set(kw.split()) & set(term.split())) / max(len(kw.split()), len(term.split()))
                            if similarity > 0.5:
                                keyword_score += X_dense[i] * similarity * 1.5
                    
                    if keyword_score > 0:
                        keyword_matches[original_kw] = keyword_score
                        theme_score += keyword_score

                # Title boost
                if title:
                    clean_title = self.clean_text(title)
                    title_boost = sum(2.0 for kw in clean_keywords if kw in clean_title)
                    theme_score *= (1.0 + 0.3 * title_boost)

                scores[theme_id] = theme_score

            # If no matches found
            if not scores:
                raise ValueError("No matching themes found for the document")

            # Find best match and calculate relative confidence
            best_theme_id = max(scores, key=scores.get)
            best_score = scores[best_theme_id]
            
            # Calculate confidence as a ratio of the best score to the sum of all scores
            total_scores = sum(scores.values())
            normalized_score = best_score / total_scores if total_scores > 0 else 0

            # Ensure score is between 0 and 1
            normalized_score = max(0.0, min(1.0, normalized_score))

            return ClassificationResult(
                theme_id=best_theme_id,
                theme_name=themes[best_theme_id][0],
                confidence_score=normalized_score,
                matching_keywords=dict(matching_keywords.get(best_theme_id, {}))
            )

        except Exception as e:
            raise ValueError(f"Classification failed: {str(e)}")
        
    def get_similar_documents(self, input_text: str, other_docs: List[Tuple[str, str]], top_n: int = 5) -> List[Tuple[int, float, str]]:
        """
        Get similar documents based on content similarity and theme classification.
        
        Args:
            input_text: The text of the document to compare
            other_docs: List of tuples containing (document_text, theme_name)
            top_n: Number of similar documents to return
            
        Returns:
            List of tuples containing (document_index, similarity_score, theme_name)
        """
        try:
            if not other_docs:
                return []
            
            # Classify the input text to get its theme
            input_theme = self.classify_single_text(input_text, self._get_theme_dict()).theme_name
            
            # Filter other_docs to only include documents with the same theme
            same_theme_docs = [
                (text, theme) for text, theme in other_docs 
                if theme == input_theme
            ]
            
            # If no documents with the same theme, return empty list
            if not same_theme_docs:
                return []
            
            # Separate documents and their themes
            doc_texts, doc_themes = zip(*same_theme_docs)
            
            # Clean all texts
            cleaned_input = self.clean_text(input_text)
            cleaned_docs = [self.clean_text(doc) for doc in doc_texts]
            
            # Calculate TF-IDF similarity
            corpus = [cleaned_input] + cleaned_docs
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            # Get top N documents
            top_indices = cosine_sim.argsort()[::-1][:top_n]
            
            # Return results with scores and themes
            return [
                (int(i), 
                float(cosine_sim[i]), 
                doc_themes[i]
                ) for i in top_indices
            ]
        except Exception as e:
            raise ValueError(f"Similarity calculation failed: {e}")

    def _get_theme_dict(self) -> Dict[str, Tuple[str, List[str]]]:
        """Helper method to get theme dictionary from database."""
        from .models import Theme  # Import here to avoid circular import
        
        themes = Theme.objects.all()
        return {
            str(theme.id): (
                theme.name,
                theme.get_fixed_keywords() + theme.get_dynamic_keywords()
            )
            for theme in themes
        }

    def calculate_top_terms(self, documents: List[str], theme_name: str) -> List[Tuple[str, float]]:
        """Calculate top TF-IDF terms for a theme."""
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