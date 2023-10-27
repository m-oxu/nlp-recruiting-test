import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
import unidecode
import re


def clean_text(text):
    # Converte o texto para minúsculas
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remover números
    # Remove acentos
    text = unidecode.unidecode(text)
    
    # Tokenização
    words = word_tokenize(text, language='portuguese')
    
    # Remove stopwords e pontuação
    stop_words = set(stopwords.words('portuguese'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # Stemming
    stemmer = RSLPStemmer()
    words = [stemmer.stem(word) for word in words]
    
    # Reconstroi o texto
    cleaned_text = ' '.join(words)
    
    return cleaned_text
