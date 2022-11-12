# Imports
import os, time, pandas, pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from TextProcessor import *
from typing import List

# Globals
default_binarizer   = MultiLabelBinarizer()
default_vectorizer  = TfidfVectorizer(max_df=0.8, max_features=10000)
default_classifier  = OneVsRestClassifier(LogisticRegression())

# Methods
def load_pickle() -> List[object]:
    pickle_file = open(f"../Resources/Pickle", "rb")
    loaded_binarizer, loaded_vectorizer, loaded_classifier = pickle.load(pickle_file)
    pickle_file.close()

    binarizer = loaded_binarizer if loaded_vectorizer is not None else default_binarizer
    vectorizer = loaded_vectorizer if loaded_vectorizer is not None else default_vectorizer
    classifier = loaded_classifier if loaded_classifier is not None else default_classifier

    return [binarizer, vectorizer, classifier]

def save_pickle(binarizer: object, vectorizer: object, classifier: object) -> None:
    # Validation
    saved_binarizer = binarizer if binarizer is not None else default_binarizer
    saved_vectorizer = vectorizer if vectorizer is not None else default_vectorizer
    saved_classifier = classifier if classifier is not None else default_classifier

    pickle_file = open(f"../Resources/Pickle", "wb")
    pickle.dump([saved_binarizer, saved_vectorizer, saved_classifier], pickle_file)
    pickle_file.close()

def train(train_screenplays: pandas.DataFrame) -> List[object]:
    # Loads classifier's variables from file
    binarizer, vectorizer, classifier = load_pickle()

    binarizer.fit(train_screenplays["Actual Genres"])
    y = binarizer.transform(train_screenplays["Actual Genres"])

    x_train, x_validation, y_train, y_validation = train_test_split(train_screenplays["Text"], y, test_size=0.2,
                                                                    random_state=42)

    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_validation_tfidf = vectorizer.transform(x_validation)

    classifier.fit(x_train_tfidf, y_train)
    # y_predictions = one_vs_rest_classifier.predict(x_validation_tfidf)

    y_probabilities = classifier.predict_proba(x_validation_tfidf)
    y_predictions = (y_probabilities >= 0.1).astype(int)
    # print(f1_score(y_validation, y_predictions, average="micro"))

    return [binarizer, vectorizer, classifier]

def classify(classifier_variables: List[object], test_screenplays: pandas.DataFrame) -> pandas.DataFrame:
    binarizer, vectorizer, classifier = classifier_variables
    classifications_dict = {}
    concordances_dict = {}
    word_appearances_dict = {}
    classifications_complete = 0

    for offset, test_screenplay in test_screenplays.iterrows():
        test_vector = vectorizer.transform([test_screenplay["Text"]])
        test_prediction = binarizer.inverse_transform(classifier.predict(test_vector))
        predicted_genres = list(sum(test_prediction, ())) + ["Unknown", "Unknown", "Unknown"] # Flattens list of tuples
        concordance, word_appearances = build_concordance_and_word_appearances(test_screenplay["Text"])

        classifications_dict[test_screenplay["Title"]] = predicted_genres[:3]
        concordances_dict[test_screenplay["Title"]] = [concordance]
        word_appearances_dict[test_screenplay["Title"]] = [word_appearances]

        classifications_complete += 1
        print(classifications_complete)

        time.sleep(0.3) # Sleep for 0.3 seconds

    # Saves classifier's variables to file
    save_pickle(binarizer, vectorizer, classifier)

    return pandas.DataFrame({"Title": classifications_dict.keys(),
                                         "Predicted Genres": classifications_dict.values(),
                                         "Concordance": concordances_dict.values(),
                                         "Word Appearances": word_appearances_dict.values()})