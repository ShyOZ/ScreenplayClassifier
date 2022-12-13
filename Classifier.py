# Imports
import numpy
import pandas, pickle

from os import listdir
from time import sleep
from pathlib import Path

from skmultilearn.adapt import MLkNN
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from Setup import *

# Methods
def load_model():
    pickle_path = Path.cwd() / "Classifier/Pickle"

    # Validation
    if not Path.exists(pickle_path):
        return train()

    # Reads model variables from pickle file
    pickle_file = open(pickle_path, "rb")
    model = pickle.load(pickle_file)
    pickle_file.close()

    return model

def save_model(model):
    # Writes model variables to pickle file
    pickle_file = open(Path.cwd() / "Classifier/Pickle", "wb")
    pickle.dump(model, pickle_file)
    pickle_file.close()

def getOptimalAmountOfNeighbors(x, t):
  hyper_params = {"n_neighbors": list(range(1, 20))}
  grid_search = GridSearchCV(KNeighborsClassifier(n_neighbors=5), hyper_params).fit(x, t)

  return grid_search.best_params_["n_neighbors"]
def getOptimalAmountOfEstinators(x, t, baseEstimator):
  hyper_params = {"n_estimators": list(range(10, 21)), "bootstrap": [True, False]}
  grid_search = GridSearchCV(BaggingClassifier(base_estimator=baseEstimator, random_state=1), hyper_params,
                            scoring="neg_log_loss").fit(x, t)

  return [grid_search.best_params[key] for key in hyper_params.keys()]

def probabilities_to_percentages(probabilities):
    probabilities_dict = dict(zip(genre_labels, probabilities))
    probabilities_dict = dict(sorted(probabilities_dict.items(), key=lambda item: item[1], reverse=True))
    sum_of_probabilities = sum(probabilities_dict.values())
    percentages_dict = {}

    # Converts each genre's probability to matching percentage
    for genre, probability in probabilities_dict.items():
        percentages_dict[genre] = (probability / sum_of_probabilities) * 100

    return percentages_dict

def train():
    train_directory = f"./TrainScreenplays/"
    train_file_paths = [train_directory + file_name for file_name in listdir(train_directory)]
    train_screenplays = pandas.merge(load_screenplays(train_file_paths), load_genres(), on="Title")

    # Creates multi-label binary representation to the screenplays' genres
    binarizer = MultiLabelBinarizer()
    binarizer.fit(train_screenplays["Actual Genres"])
    y = binarizer.transform(train_screenplays["Actual Genres"])

    # Splits screenplays collection to train and validation
    x_train, x_validation, y_train, y_validation = train_test_split(train_screenplays["Text"].values, y,
                                                                    test_size=0.2, random_state=1000)

    # Extract features from train screenplays
    vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
    x_train = vectorizer.fit_transform(x_train)
    x_validation = vectorizer.transform(x_validation)

    # Classifies the test screenplays
    '''
        CLASSIFIERS HISTORY:
        OneVsRestClassifier(LogisticRegression()) -> 0.1084
    '''

    # TODO: FIX
    parameters = {"k": range(1, 20), "s": [0.5, 0.7, 1.0]}
    classifier = GridSearchCV(MLkNN(), parameters, scoring="f1_macro")
    classifier.fit(x_train, y_train)
    score = classifier.score(x_validation, y_validation)
    print("Accuracy: {:.4f}".format(score))

    # Saves model variables to file
    save_model([vectorizer, classifier])

    return [vectorizer, classifier]

def classify(model, test_screenplays):
    vectorizer, classifier = model
    classifications_dict = {}
    classifications_complete = 0

    for offset, test_screenplay in test_screenplays.iterrows():
        test_vector = vectorizer.transform([test_screenplay["Text"]])
        test_probabilities = sum(classifier.predict_proba(test_vector).tolist(), []) # Flattens the list
        test_percentages = probabilities_to_percentages(test_probabilities)

        classifications_dict[test_screenplay["Title"]] = test_percentages

        # Prints progress (for GUI to update progress)
        classifications_complete += 1
        print(classifications_complete)

        sleep(0.5) # Sleep for 0.5 seconds

    return pandas.DataFrame({"Title": classifications_dict.keys(),
                             "GenrePercentages": classifications_dict.values()})