import re
import PyPDF2
import docx
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from nlp_id.lemmatizer import Lemmatizer
from indoNLP.preprocessing import replace_word_elongation
from indoNLP.preprocessing import replace_slang
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

nltk.download('punkt')
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('indonesian')
lemmatizer = Lemmatizer()

# 1. Extract text dari files
def extract_pdf_text(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_docx_text(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

# 2. Extract sections
def extract_sections(docx_text):
    text = re.sub(r'\s+', ' ', docx_text).strip()
    patterns = {
        "title": (
            r'(?im)^(?:[A-Z][A-Z\s]+\n)?'
            r'([^\n]+?(?=\s*(?:ABSTRAK|ABSTRACT|SKRIPSI|PROGRAM STUDI)))'
        ),
        "abstract": (
            r'(?im)(?:ABSTRAK|ABSTRACT)[\s.:-]*\s*'
            r'([^\n].*?)\s*'
            r'(?=(?:Kata Kunci|Keywords|Latar Belakang|BAB I|1\. PENDAHULUAN))'
        ),
        "background": (
            r'(Latar Belakang[\s\S]+?)(?=(?:Perumusan Masalah|Rumusan Masalah))'
        ),
        "conclusion": (
            r'(Kesimpulan[\s\S]+?)(?=(?:Saran|Daftar Pustaka))'
        )
    }

    results = {}
    for section, pattern in patterns.items():
        try:
            if section == "conclusion":
                matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                content = matches[-1] if matches else "Not found"
            elif section == "background":
                matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                content = matches[1] if len(matches) > 1 else "Not found"
            else:
                match = re.search(pattern, text, re.DOTALL)
                content = match.group(1) if match else "Not found"
                if section == "title" and match:
                    content = match.group(0)

            if content != "Not found":
                content = re.sub(r'\s+', ' ', content).strip()
                if len(content) > 10:
                    content = re.sub(r'^\d+\.\d+\s*', '', content)
                    content = re.sub(r'\s{2,}', ' ', content)
                    results[section] = content
                else:
                    results[section] = "Not found (content too short)"
            else:
                results[section] = "Not found"
        except Exception as e:
            results[section] = f"Error in extraction: {e}"
    return results

# 3. Text cleaningz
def clean_text_pipeline(title, abstract, background, conclusion):
    combined_text = f"{title} {abstract} {background} {conclusion}".strip()
    try:
        lemmatized = lemmatizer.lemmatize(combined_text)
    except Exception:
        lemmatized = combined_text

    text_processed = replace_word_elongation(lemmatized)
    tokens = nltk.word_tokenize(text_processed)
    tokens = [t for t in tokens if t.isalpha()]
    filtered_tokens = [t for t in tokens if t not in stopwords]

    return " ".join(filtered_tokens)

# 4. Classify satu text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def classify_single_text(document, themes, title=None):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Include bigrams as well as unigrams

    # Vectorize the document text
    X = vectorizer.fit_transform([document])
    feature_names = vectorizer.get_feature_names_out()
    X_dense = X.toarray()[0]  # Get the TF-IDF vector for the document

    scores = {}

    # Loop through each theme and calculate score based on keyword matching
    for cid, (name, kws) in themes.items():
        score = 0
        # Iterate through each keyword in the theme
        for kw in kws:
            # Loop through each feature in the document vector
            for i, term in enumerate(feature_names):
                # Match the keyword with the term and accumulate the score
                if kw.lower() in term.lower():  # Case-insensitive matching
                    score += X_dense[i]  # Add the TF-IDF score for this term

        # Store the score for this theme
        scores[cid] = score

    # Find the theme with the highest score
    best = max(scores, key=scores.get)

    return {
        'theme_id': best,
        'theme_name': themes[best][0],  # Return the theme name with the highest score
        'confidence_score': scores[best]  # The accumulated score
    }


# 5. Similar documents
def get_similar_documents(input_text, other_docs, top_n=5):
    corpus = [input_text] + other_docs
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    top_indices = cosine_sim.argsort()[::-1][:top_n]
    return [(i, cosine_sim[i]) for i in top_indices]

# 6. Calculate top TF-IDF Kata Kunci untuk Tiap KK
def calculate_top_tfidf_terms(documents):
    # Extract cleaned texts and themes from documents
    texts = [doc.cleaned_text for doc in documents]
    themes = [doc.theme for doc in documents]

    # Mapping themes to texts
    theme_texts = defaultdict(list)
    for text, theme in zip(texts, themes):
        theme_texts[theme].append(text)

    # Dictionary to store the top terms for each theme
    top_terms_per_theme = {}

    # Calculate TF-IDF for each theme
    for theme, texts in theme_texts.items():
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        
        # Create a dictionary of term scores
        term_scores = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names))}
        
        # Get top 5 terms based on scores
        top_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        top_terms_per_theme[theme] = top_terms

    return top_terms_per_theme