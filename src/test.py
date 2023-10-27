import pickle
from preprocess import preprocess
from train import train
import pandas as pd

def test(test_data):
    model = pickle.load("models/lr_tfidf_binary.pkl")

    test_preprocessed, y = preprocess(test_data, train=True)
    pred_results = model.predict(test_preprocessed)
    print(classification_report(y, pred_results))

if "__main__":
    is_treino = input("Você está treinando o modelo? (y/n): ")
    df = pd.read_csv("data/train_df.csv")

    if is_treino == 'y':
        train(df)

        test(pd.read_csv("data/test_df.csv"))