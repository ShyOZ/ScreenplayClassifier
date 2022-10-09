# Imports
import time
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from TextProcessor import *
from tqdm import tqdm
from typing import Tuple

# Globals
multilabel_binarizer = MultiLabelBinarizer()
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
one_vs_rest_classifier = OneVsRestClassifier(LogisticRegression())
module_trained = False

# Methods
def train(train_screenplays: pd.DataFrame) -> None:
    multilabel_binarizer.fit(train_screenplays["Actual Genres"])
    y = multilabel_binarizer.transform(train_screenplays["Actual Genres"])

    x_train, x_validation, y_train, y_validation = train_test_split(train_screenplays["Text"], y, test_size=0.2,
                                                                    random_state=42)

    x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
    x_validation_tfidf = tfidf_vectorizer.transform(x_validation)

    one_vs_rest_classifier.fit(x_train_tfidf, y_train)
    # y_predictions = one_vs_rest_classifier.predict(x_validation_tfidf)

    y_probabilities = one_vs_rest_classifier.predict_proba(x_validation_tfidf)
    y_predictions = (y_probabilities >= 0.1).astype(int)
    # print(f1_score(y_validation, y_predictions, average="micro"))

    module_trained = True

def classify(train_screenplays: pd.DataFrame, test_screenplays : pd.DataFrame) -> pd.DataFrame:
    classifications_dict = {}
    concordances_dict = {}
    word_appearances_dict = {}
    progress_bar = tqdm(test_screenplays.iterrows(), total=test_screenplays.shape[0], disable=True)

    for offset, test_screenplay in progress_bar:
        test_vector = tfidf_vectorizer.transform([test_screenplay["Text"]])
        test_prediction = multilabel_binarizer.inverse_transform(one_vs_rest_classifier.predict(test_vector))
        predicted_genres = list(sum(test_prediction, ())) + ["Unknown", "Unknown", "Unknown"] # Flattens list of tuples
        concordance, word_appearances = build_concordance_and_word_appearances(test_screenplay["Text"])

        classifications_dict[test_screenplay["Title"]] = predicted_genres[:3]
        concordances_dict[test_screenplay["Title"]] = [concordance]
        word_appearances_dict[test_screenplay["Title"]] = [word_appearances]

        # TODO: COMPLETE (Prints progress bar's percent (for WPF GUI))

    return pd.DataFrame({"Title": classifications_dict.keys(),
                                         "Predicted Genres": classifications_dict.values(),
                                         "Concordance": concordances_dict.values(),
                                         "Word Appearances": word_appearances_dict.values()})