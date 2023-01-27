# Imports
import spacy, re, datetime

from collections import Counter
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from textblob import TextBlob
from transformers import pipeline

# Globals
from Loader import genre_labels

emotions_pipeline = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=100)
emotion_labels = ["Anger", "Fear", "Joy", "Love", "Sadness",  "Surprise"]

zero_shot_pipeline = pipeline("zero-shot-classification", model="oigele/Fb_improved_zeroshot")

entities_pipeline = spacy.load("en_core_web_sm")

# Methods
def get_dominant_time_of_day(text):
    daytime_counter, nighttime_counter = 0, 0
    dusk_time, dawn_time = datetime.time(18), datetime.time(6)

    # Checks all time expressions in the text
    time_expressions = re.findall(r"^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$", text)
    time_expressions = [datetime.datetime.strptime(time_str, "%H:%M").time() for time_str in time_expressions]

    daytime_counter += len([time for time in time_expressions if dawn_time <= time < dusk_time])
    nighttime_counter += len(time_expressions) - daytime_counter

    # Checks all day/night related expressions in the text
    daytime_expressions = ["day", "dawn", "morning", "noon", "sun"]
    days_of_week = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
    nighttime_expressions = ["night", "dusk", "evening", "twilight"]
    words = clean_words(TextBlob(text).words)

    for expression in daytime_expressions:
        daytime_counter += len([w for w in words if (expression in w) and (w not in days_of_week)])
    for expression in nighttime_expressions:
        nighttime_counter += len([w for w in words if expression in w])

    return {"Dominant Time of Day": "Daytime" if daytime_counter > nighttime_counter else "Nighttime"}

def get_setting(text):
    past_counter, present_counter, future_counter = 0, 0, 0

    # Retrieves the time period of the setting
    years = [int(year) for year in re.findall(r"[0-9]{4}", text)]
    past_counter += len([year for year in years if year < 1990])
    future_counter += len([year for year in years if year > datetime.date.today().year])
    past_counter += len(years) - (past_counter + future_counter)

    time_period = "Present"
    if all(past_counter > counter for counter in [present_counter, future_counter]):
        time_period = "Past"
    elif all(present_counter > counter for counter in [past_counter, future_counter]):
        time_period = "Present"
    elif all(future_counter > counter for counter in [past_counter, present_counter]):
        time_period = "Future"

    # Retrieves the most common locations of the setting
    locations = clean_words([ent.text for ent in entities_pipeline(text).ents if ent.label_ in ["GPE", "LOC", "FAC"]])
    locations = [location[0] for location in Counter(locations).most_common(5)]

    return {"Time Period": time_period, "Locations": locations}

def clean_words(words):
    # Cleans the words from punctuation
    words = [re.sub(r"[^\w\s]", "", w) for w in words]

    # Cleans the words from stopwords
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w.lower() not in stop_words]

    return words

def get_protagonist_role(text):
    # Extracts protagonist's actions from text
    blob, protagonist = TextBlob(text), get_protagonist(text)
    actions = " ".join([str(sentence) for sentence in blob.sentences if sentence.startswith(protagonist + " ")])

    # Predicts the protagonist's role by his/her actions
    protagonist_roles_dict = {"Action": "Fighter", "Adventure": "Traveller/Explorer", "Comedy": "Comedian",
                              "Crime": "Law Person/Outlaw", "Drama": "Realist", "Family": "Child",
                              "Fantasy": "Creature/Royalty/Knight/Sorcerer", "Horror": "Monster/Killer/Victim",
                              "Romance": "Spouse/Lover/Ex", "SciFi": "Superhuman/Scientist/Alien/Robot",
                              "Thriller": "Detective/Spy", "War": "Soldier"}

    action_genres = ["Drama"] if len(actions) == 0 \
        else zero_shot_pipeline(actions, genre_labels, multi_label=True)["labels"]

    return {"Protagonist Roles": protagonist_roles_dict[action_genres[0]]}

def get_protagonist(text):
    # Extracts character names from text
    blob = TextBlob(text)
    proper_nouns = [word[0] for word in blob.pos_tags if word[1] == "NNP"]
    entities = entities_pipeline(",".join(proper_nouns)).ents
    characters = [entity.text for entity in entities if "," not in entity.text and entity.label_ == "PERSON"]
    characters = set([word for word in characters if word.isupper()])

    # Counts appearances for each character (the protagonist is the character which appears the most)
    characters_dict = dict(zip(characters, [blob.word_counts[character.lower()] for character in characters]))
    characters_dict = dict(sorted(characters_dict.items(), key=lambda item: item[1], reverse=True))
    protagonist = list(characters_dict.keys())[0]

    return protagonist.capitalize()

def get_emotions(text):
    # Organizes emotions and their percentages in dictionary
    emotions_scores = sum(emotions_pipeline(text, truncation=True), [])  # Flattens the list
    keys = [emotion_label for emotion_label in emotion_labels]
    emotion_scores_dict = dict(zip(keys, [0 for key in keys]))

    for emotion in emotions_scores:
        emotion_scores_dict[emotion["label"].capitalize()] = emotion["score"]

    return emotion_scores_dict

def get_topics(text):
    # Models the text's topics
    sentences = TextBlob(text).sentences
    tokens = [b for l in sentences for b in zip(l.split(" ")[:-1], l.split(" ")[1:])] # e.g: Day + Time = Daytime
    dictionary = corpora.Dictionary(tokens)
    bag_of_words = [dictionary.doc2bow(t) for t in tokens]
    lda_model = LdaModel(bag_of_words, num_topics=3, id2word=dictionary, passes=10, chunksize=10,
                         update_every=1, alpha="auto", per_word_topics=True, random_state=42)
    topic_descriptives = " ".join([re.sub(r"[^a-zA-Z\s]", "", topic) for offset, topic in lda_model.print_topics()])

    # Retrieves the text main topic by its descriptives
    genre_topics_dict = {"Action": ["Violence", "Danger", "Chase"], "Adventure": ["Exploration", "Journey"],
                         "Comedy": ["Amusement", "Laughter"], "Crime": ["Crime", "Trial"],
                         "Drama": ["Conflict", "Crisis", "Tragedy"], "Family": ["Adolescence", "Care", "Friendship"],
                         "Fantasy": ["Supernatural", "Magic", "Wonder"],
                         "Horror": ["Terror", "Disgust", "Murder", "Gore"],
                         "Romance": ["Emotion", "Lust", "Relationship"],
                         "SciFi": ["Science", "Technology", "Space", "Future"],
                         "Thriller": ["Espionage", "Conspiracy", "Cunning"], "War": ["War", "Death", "Politics"]}
    topic_genres = ["Drama"] if len(topic_descriptives) == 0 \
        else zero_shot_pipeline(topic_descriptives, genre_labels, multi_label=True)["labels"]

    return {"Topics": genre_topics_dict[topic_genres[0]]}

def extract_features(screenplay_title, screenplay_text):
    # Extracts features by the screenplay text
    features_dict = {"Title": screenplay_title}
    features_dict.update(get_topics(screenplay_text))
    features_dict.update(get_emotions(screenplay_text))
    features_dict.update(get_protagonist_role(screenplay_text))
    features_dict.update(get_setting(screenplay_text))
    features_dict.update(get_dominant_time_of_day(screenplay_text))

    return features_dict