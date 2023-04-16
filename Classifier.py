# Imports
import time
import numpy
import pandas
import pickle
import Constants

from Loader import load_test_screenplays
#from ScreenplayProcessor import time_periods, times_of_day
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer

from ScreenplayProcessor import TOKENIZER, MAX_SEQUENCE_LENGTH, create_nn_model

# Methods
def load_model():
    # Validates existence of pickle file
    if not Constants.model_pickle_path.exists():
        return create_nn_model() # create_ml_model()

    # Reads model variables from pickle file
    pickle_file = open(Constants.model_pickle_path, "rb")
    model = pickle.load(pickle_file)
    pickle_file.close()

    return model

def save_model(model):
    # Writes model variables to pickle file
    pickle_file = open(Constants.model_pickle_path, "wb")
    pickle.dump(model, pickle_file)
    pickle_file.close()

# def encode(screenplays):
#     # Encodes the textual values in the screenplays dataframe
#     screenplays.replace({"Time Period": {period: time_periods.index(period) for period in time_periods},
#                          "Time of Day": {time: times_of_day.index(time) for time in times_of_day}},
#                         inplace=True)
#
#     return screenplays

def get_best_amount_of_neighbors(x, t):
  hyper_params = {"n_neighbors": list(range(1, 20))}
  grid_search = GridSearchCV(KNeighborsClassifier(n_neighbors=5), hyper_params).fit(x, t)

  return grid_search.best_params_["n_neighbors"]

def get_best_amount_of_estimators(x, t, base_estimator):
  hyper_params = {"n_estimators": list(range(10, 21)), "bootstrap": [True, False]}
  grid_search = GridSearchCV(BaggingClassifier(base_estimator=base_estimator, random_state=1), hyper_params,
                             scoring="neg_log_loss").fit(x, t)

  return grid_search.best_params_["n_estimators"], grid_search.best_params_["bootstrap"]

def create_ml_model():
    # Creates a classification model
    train_screenplays = pandas.read_csv(Constants.train_csv_path)
    train_screenplays["Genres"] = [eval(genres) for genres in train_screenplays["Genres"]]

    t = MultiLabelBinarizer().fit_transform(train_screenplays["Genres"])
    x = train_screenplays.drop(["Title", "Genres"], axis=1)
    x_train, x_validation, t_train, t_validation = train_test_split(x, t, test_size=0.2, random_state=42)

    # Builds classifier and predicts its accuracy score (current best: SVC(probability=True) - 0.1441)
    base_estimator = KNeighborsClassifier(n_neighbors=get_best_amount_of_neighbors).fit(x, t)
    amount_of_estimators, using_bootstrap = get_best_amount_of_estimators(x, t, base_estimator)
    ensembles_classifier = BaggingClassifier(base_estimator=base_estimator, n_estimators=amount_of_estimators,
                                             bootstrap=using_bootstrap, random_state=1).fit(x, t)

    classifier = MultiOutputClassifier(ensembles_classifier)
    score = classifier.score(x, t).mean()
    print("Accuracy: {:.4f}".format(score))

    # Saves the model to file
    save_model(classifier)

    return classifier

def probabilities_to_percentages(probabilities):
    # Creates a sorted probabilities dictionary
    probabilities_dict = dict(zip(Constants.genre_labels, probabilities))
    probabilities_dict = dict(sorted(probabilities_dict.items(), key=lambda item: item[1], reverse=True))
    sum_of_probabilities = sum(probabilities)
    percentages_dict = {}

    # Converts each genre's probability to matching percentage
    for genre, probability in probabilities_dict.items():
        percentages_dict[genre] = (probability / sum_of_probabilities) * 100

    return percentages_dict

def classify(file_paths):
    # Loads test screenplays and classification model
    test_screenplays = load_test_screenplays(file_paths)
    model = load_model()
    classifications_dict = {}
    classifications_complete = 0

    # Tokenizes the screenplays' texts (NN way)
    sequences = TOKENIZER.texts_to_sequences(test_screenplays["Text"])
    sequences_matrix = sequences.pad_sequences(sequences, max_len=MAX_SEQUENCE_LENGTH)
    predictions = pandas.DataFrame(model.predict(sequences_matrix), columns=Constants.genre_labels)

    # Classifies the test screenplays
    for offset, screenplay in test_screenplays.iterrows():
        # features_vector = [feature[0] for feature in numpy.array(screenplay[1:]).reshape(-1, 1)]
        # genre_probabilities = [probability.tolist()[0] for probability in classifier.predict_proba([features_vector])]
        # genre_probabilities = [probability[0] for probability in genre_probabilities]
        classifications_dict[file_paths[offset]] = probabilities_to_percentages(predictions[offset])

        # Prints progress (for GUI to update progress)
        classifications_complete += 1
        print(classifications_complete)

        time.sleep(0.5)  # seconds

    return pandas.DataFrame({"FilePath": classifications_dict.keys(),
                             "GenrePercentages": classifications_dict.values()})