# Imports
import numpy, pandas, pickle, os, time, pathlib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

import Loader, ScreenplayProcessor
from Loader import *
from ScreenplayProcessor import *

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

def encode(screenplays):
    # Encodes the textual values in the screenplays dataframe
    topics = list(ScreenplayProcessor.genre_topics_dict.values())
    protagonist_roles = list(ScreenplayProcessor.protagonist_roles_dict.values())

    screenplays.replace({"Topics": {topic: topics.index(topic) for topic in topics},
                          "Protagonist Roles": {role: protagonist_roles.index(role) for role in protagonist_roles},
                          "Time Period": {period: ScreenplayProcessor.time_periods.index(period)
                                          for period in ScreenplayProcessor.time_periods},
                          "Dominant Time of Day": {time: ScreenplayProcessor.times_of_day.index(time)
                                                   for time in ScreenplayProcessor.times_of_day}},
                        inplace=True)

    return screenplays

def create_model():
    # Creates a classification model
    train_screenplays = encode(pandas.read_csv(".\Classifier\Train.csv"))

    t = MultiLabelBinarizer().fit_transform(train_screenplays["Genres"])
    x = train_screenplays.drop(["Title", "Genres"], axis=1)

    x_train, x_validation, y_train, y_validation = train_test_split(x, t, test_size=0.2, random_state=1)

    # Builds classifier and predicts its accuracy score (best score: 0.0)
    # TOOD: IMPROVE THE base model with hyper_parameters
    base_model = OneVsRestClassifier(LogisticRegression())
    classifier = base_model.fit(x_train, y_train)

    y_predictions = classifier.predict(x_validation)
    score = accuracy_score(y_validation, y_predictions)
    print("Accuracy: {:.4f}".format(score))

    # Saves model variables to file
    save_model(classifier)

    return classifier

def probabilities_to_percentages(probabilities):
    probabilities_dict = dict(zip(Loader.genre_labels, probabilities))
    probabilities_dict = dict(sorted(probabilities_dict.items(), key=lambda item: item[1], reverse=True))
    sum_of_probabilities = sum(probabilities)
    percentages_dict = {}

    # Converts each genre's probability to matching percentage
    for genre, probability in probabilities_dict.items():
        percentages_dict[genre] = (probability / sum_of_probabilities) * 100

    return percentages_dict

def classify(screenplays):
    # Loads classification model
    screenplays = encode(screenplays)
    classifier = load_model()
    classifications_dict = {}
    classifications_complete = 0

    # TODO: Figure out the test_probabilities
    for offset, screenplay in screenplays.iterrows():
        features_vector = [x[0] for x in numpy.array(screenplay[1:]).reshape(-1, 1)]
        test_probabilities = classifier.predict_proba([features_vector])
        print(test_probabilities)

        # test_percentages = probabilities_to_percentages(test_probabilities)
        # classifications_dict[screenplay["Title"]] = test_percentages

        # Prints progress (for GUI to update progress)
        # classifications_complete += 1
        # print(classifications_complete)

        time.sleep(0.5) # seconds

    return pandas.DataFrame({"Title": classifications_dict.keys(),
                             "GenrePercentages": classifications_dict.values()})