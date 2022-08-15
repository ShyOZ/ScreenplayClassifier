# Imports
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from typing import List, Dict

# Methods
def classify(screenplay_dataframe: pd.DataFrame, screenplay_names : List[str]) -> List[str]:
    genre_labels = open("../Resources/Genres.txt").read().splitlines()
    x, t = screenplay_dataframe["Text"], screenplay_dataframe[genre_labels]

    x_train, x_validation, y_train, y_validation = train_test_split(x, t, test_size=0.2, random_state=42)
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
    x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
    x_validation_tfidf = tfidf_vectorizer.transform(x_validation)

    one_vs_rest_classifier = OneVsRestClassifier(LogisticRegression())
    one_vs_rest_classifier.fit(x_train_tfidf, y_train)
    y_predictions = one_vs_rest_classifier.predict(x_validation_tfidf)

    print(y_predictions)

    return ["", "", ""]