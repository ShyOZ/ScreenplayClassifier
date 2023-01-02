# Imports
import re, pandas

from spacy import load
from transformers import pipeline

# Globals
EMOTIONS = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None)
emotion_labels = ["Sadness", "Anger", "Fear", "Joy", "Surprise", "Love"]

NER = load("en_core_web_sm")

# Methods
def get_screenplay_emotions(screenplay_text):
    emotions_scores = sum(EMOTIONS(screenplay_text, truncation=True), [])  # Flattens the list
    emotions_dict = {}

    # Organizes emotions and their scores in dictionary
    for emotion in emotions_scores:
        emotions_dict[emotion["label"].capitalize()] = emotion["score"]

    return emotions_dict

def get_screenplay_entities(screenplay_text):
    entities = NER(screenplay_text).ents
    entity_labels = set([entity.label_ for entity in entities])
    entities_dict = {}

    # Organizes entities in dictionary
    for entity_label in entity_labels:
        entities_dict[entity_label] = set([entity.text for entity in entities if entity.label_ == entity_label])

    return entities_dict

def extract_features(screenplay_title, screenplay_text):
    features_dict = {"Title": screenplay_title}

    # TODO: FURTHER FEATURE EXTRACTION (IF NECESSARY)
    features_dict.update(get_screenplay_emotions(screenplay_text))
    # features_dict.update(get_screenplay_entities(screenplay_text))

    return features_dict

