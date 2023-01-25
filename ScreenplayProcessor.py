# Imports
import re, pandas

from nltk import sent_tokenize
from spacy import load
from transformers import pipeline

# Globals
EMOTIONS = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None)
emotion_labels = ["Sadness", "Anger", "Fear", "Joy", "Surprise", "Love"]

# SENTIMENT = pipeline("sentiment-analysis", model="distilbert-base-uncased")
# sentiment_labels = ["Negative", "Positive"]

# NER = load("en_core_web_sm")

# Methods
def get_screenplay_emotions(screenplay_text):
    emotions_scores = sum(EMOTIONS(screenplay_text, truncation=True), [])  # Flattens the list
    emotions_dict = {}

    # Organizes emotions and their scores in dictionary
    for emotion in emotions_scores:
        emotions_dict[emotion["label"].capitalize()] = emotion["score"]

    return emotions_dict

def get_screenplay_sentiment(screenplay_text):
    pass
    # screenplay_sentences = sent_tokenize(screenplay_text)
    # sentence_sentiments = [SENTIMENT(sentence)["label"] for sentence in screenplay_sentences]

    # return max(sentence_sentiments.count(sentiment_labels[0]), sentence_sentiments.count(sentiment_labels[0]))

def extract_features(screenplay_title, screenplay_text):
    features_dict = {"Title": screenplay_title}

    # TODO: FURTHER FEATURE EXTRACTION (IF NECESSARY)
    features_dict.update(get_screenplay_emotions(screenplay_text))
    # features_dict.update(get_screenplay_sentiment(screenplay_text))

    return features_dict

