from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
from preprocess import preprocess

def train(df):
    model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')

    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit and predict
    model.fit(X_train, y_train)

    pickle.dump("models/lr_tfidf_binary.pkl")