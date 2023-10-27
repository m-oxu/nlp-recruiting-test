from utils import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def preprocess(df, train=True):
    df["overall_rating_binary"] = df["overall_rating"].apply(lambda rating: 0 if rating <= 3 else 1)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))

    X = tfidf_vectorizer.fit_transform(df['review_text'].apply(clean_text))
    y = df["overall_rating_binary"]

    if train:
        return X, y
    return X
