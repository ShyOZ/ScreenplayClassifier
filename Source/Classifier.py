# Imports
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from typing import Tuple

# Methods
def train(train_screenplays: pd.DataFrame) -> Tuple:
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(train_screenplays["Actual Genres"])
    y = multilabel_binarizer.transform(train_screenplays["Actual Genres"])

    x_train, x_validation, y_train, y_validation = train_test_split(train_screenplays["Text"], y, test_size=0.2,
                                                                    random_state=42)
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
    x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
    x_validation_tfidf = tfidf_vectorizer.transform(x_validation)

    one_vs_rest_classifier = OneVsRestClassifier(LogisticRegression())
    one_vs_rest_classifier.fit(x_train_tfidf, y_train)
    # y_predictions = one_vs_rest_classifier.predict(x_validation_tfidf)

    y_probabilities = one_vs_rest_classifier.predict_proba(x_validation_tfidf)
    y_predictions = (y_probabilities >= 0.1).astype(int)
    print(f1_score(y_validation, y_predictions, average="micro"))

    return tuple([multilabel_binarizer, tfidf_vectorizer, one_vs_rest_classifier])

def classify(train_screenplays: pd.DataFrame, test_screenplays : pd.DataFrame) -> pd.DataFrame:
    multilabel_binarizer, tfidf_vectorizer, one_vs_rest_classifier = train(train_screenplays)
    classifications_dict = {}

    for offset, test_screenplay in test_screenplays.iterrows():
        test_vector = tfidf_vectorizer.transform([test_screenplay["Text"]])
        test_prediction = multilabel_binarizer.inverse_transform(one_vs_rest_classifier.predict(test_vector))
        predicted_genres = list(sum(test_prediction, ())) # Flattens the list of tuples
        classifications_dict[test_screenplay["Title"]] = predicted_genres

    test_classifications = pd.DataFrame({"Title": classifications_dict.keys(),
                                         "Predicted Genres": classifications_dict.values()})

    return pd.merge(test_screenplays, test_classifications, on="Title")