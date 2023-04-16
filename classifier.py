# Imports
import time
import numpy
import pandas as pd
import pickle
import constants

from loader import load_test_screenplays
# from ScreenplayProcessor import time_periods, times_of_day
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer

from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras_preprocessing import sequence
from keras_preprocessing.text import Tokenizer

from script_loader import ScriptLoader
from script_info import ScriptInfo

# Globals
MAX_WORDS, MAX_SEQUENCE_LENGTH = 200000, 150
TOKENIZER = Tokenizer(num_words=MAX_WORDS)
CLASS_NUM = len(constants.genre_labels)
BATCH_SIZE, EPOCH = 128, 15

VALIDATION_SPLIT = 0.2


# Methods
def load_ml_model():
    # Validates existence of pickle file
    if not constants.model_pickle_path.exists():
        return create_nn_model()  # create_ml_model()

    # Reads model variables from pickle file
    pickle_file = open(constants.model_pickle_path, "rb")
    model = pickle.load(pickle_file)
    pickle_file.close()

    return model


def save_ml_model(model):
    # Writes model variables to pickle file
    pickle_file = open(constants.model_pickle_path, "wb")
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
    train_screenplays = pd.read_csv(constants.train_csv_path)
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
    score = classifier.score(x, t)
    print("Accuracy: {:.4f}".format(score))

    # Saves the model to file
    save_ml_model(classifier)

    return classifier


def create_nn_model():

    scripts_iter = ScriptLoader(constants.train_screenplays_directory,
                                ScriptInfo.schema().loads(constants.movie_info_path.read_text(), many=True),
                                True)
    data = ((text, list(genres)) for text, _, _, genres in scripts_iter)
    screenplays_texts, y = zip(*data)

    y = pd.DataFrame(MultiLabelBinarizer().fit_transform(y))

    # train_screenplays = pd.read_csv(constants.train_csv_path)
    # # Tokenizes the screenplays' texts
    # screenplays_texts = train_screenplays["Text"]
    #
    # y = pd.get_dummies(train_screenplays["Genres"])

    TOKENIZER.fit_on_texts(screenplays_texts)
    sequences = TOKENIZER.texts_to_sequences(screenplays_texts)
    sequences_matrix = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # tfidf_matrix = TOKENIZER.texts_to_matrix(screenplays_texts, mode="tfidf")
    # print(tfidf_matrix.shape)

    # Builds a Recurrent Neural Network (RNN)
    inputs = Input(name="inputs", shape=[MAX_SEQUENCE_LENGTH])
    layer = Embedding(MAX_WORDS, 50, input_length=MAX_SEQUENCE_LENGTH)(inputs)
    layer = LSTM(100)(layer)
    layer = Dense(256, activation="elu")(layer)
    layer = Dropout(0.5)(layer)
    outputs = Dense(CLASS_NUM, activation="sigmoid", name="outputs")(layer)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss="categorical_crossentropy", optimizer="Adamax", metrics="accuracy")
    model.fit(sequences_matrix, y, batch_size=BATCH_SIZE, epochs=EPOCH, validation_split=VALIDATION_SPLIT)

    model.save(constants.model_path,
               save_traces=False,)

    return model


def probabilities_to_percentages(probabilities):
    # Creates a sorted probabilities dictionary
    probabilities_dict = dict(zip(constants.genre_labels, probabilities))
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
    model = load_ml_model()
    classifications_dict = {}
    classifications_complete = 0

    # Tokenizes the screenplays' texts (NN way)
    sequences = TOKENIZER.texts_to_sequences(test_screenplays["Text"])
    sequences_matrix = sequences.pad_sequences(sequences, max_len=MAX_SEQUENCE_LENGTH)
    predictions = pd.DataFrame(model.predict(sequences_matrix), columns=constants.genre_labels)

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

    return pd.DataFrame({"FilePath": classifications_dict.keys(),
                             "GenrePercentages": classifications_dict.values()})
