# Imports
import numpy as np
import pandas, pickle, os, time, pathlib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

import Loader
from Loader import *

# Methods
def load_model():
    # Validates existence of pickle file
    pickle_path = pathlib.Path.cwd() / "Classifier/Model.pkl"
    if not pathlib.Path.exists(pickle_path):
        return create_model()

    # Reads model variables from pickle file
    pickle_file = open(pickle_path, "rb")
    model = pickle.load(pickle_file)
    pickle_file.close()

    return model

def save_model(model):
    # Writes model variables to pickle file
    pickle_file = open(pathlib.Path.cwd() / "Classifier/Model.pkl", "wb")
    pickle.dump(model, pickle_file)
    pickle_file.close()

def create_model():
    # Creates a classification model
    train_screenplays = pandas.read_csv(".\Classifier\Train.csv")

    binarizer = MultiLabelBinarizer()
    for header in ["Topics", "Protagonist Roles", "Time Period", "Dominant Time of Day", "Actual Genres"]:
        train_screenplays[header] = binarizer.fit_transform(train_screenplays[header])

    t = train_screenplays["Actual Genres"]
    x = train_screenplays.drop(["Title", "Actual Genres"], axis=1)
    x_train, x_validation, y_train, y_validation = train_test_split(x, t, test_size=0.2, random_state=1)

    # Builds classifier and predicts its accuracy score (best score: 0.7948)
    base_model = OneVsRestClassifier(LogisticRegression())
    classifier = base_model.fit(x_train, y_train)

    y_predictions = classifier.predict(x_validation)
    score = accuracy_score(y_validation, y_predictions)
    print("Accuracy: {:.4f}".format(score))

    # Saves model variables to file
    save_model(classifier)

    return classifier

def probabilities_to_percentages(probabilities):
    probabilities_dict = dict(zip(genre_labels, probabilities))
    probabilities_dict = dict(sorted(probabilities_dict.items(), key=lambda item: item[1], reverse=True))
    sum_of_probabilities = sum(probabilities)
    percentages_dict = {}

    # Converts each genre's probability to matching percentage
    for genre, probability in probabilities_dict.items():
        percentages_dict[genre] = (probability / sum_of_probabilities) * 100

    return percentages_dict

def classify(screenplays):
    # Loads classification model
    classifier = load_model()
    classifications_dict = {}
    classifications_complete = 0

    # Classifies each screenplay and organizes in dictionary
    binarizer = MultiLabelBinarizer()

    print(screenplays)
    # for offset, screenplay in screenplays.iterrows():
    #     for header in ["Topics", "Protagonist Roles", "Time Period", "Dominant Time of Day"]:
    #         screenplay[header] = binarizer.fit_transform(screenplays[header])[0]
    #     test_vector = screenplay.values[1:]
    #     print(test_vector)
        # test_probabilities = classifier.predict_proba(test_vector)
        # test_percentages = probabilities_to_percentages(test_probabilities)

        # classifications_dict[screenplay["Title"]] = test_percentages

        # Prints progress (for GUI to update progress)
        # classifications_complete += 1
        # print(classifications_complete)
        #
        # time.sleep(0.5) # seconds

    return pandas.DataFrame({"Title": classifications_dict.keys(),
                             "GenrePercentages": classifications_dict.values()})