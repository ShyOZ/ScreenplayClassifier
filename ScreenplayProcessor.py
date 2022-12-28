# Imports
import re, pandas

from concurrent.futures import ThreadPoolExecutor, as_completed
from nltk import sent_tokenize
from tqdm import tqdm
from transformers import pipeline

emotion_pipeline = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
emotion_labels = ["Sadness", "Anger", "Fear", "Joy", "Surprise", "Love"]

# Methods
def get_screenplay_emotions(screenplay_text):
    screenplay_sentences = sent_tokenize(screenplay_text)
    sentences_count = len(screenplay_sentences)
    screenplay_emotions_dict = dict(zip(emotion_labels, [0 for emotion_label in emotion_labels]))
    sentences_emotions = []

    print("\tProcessing screenplay...", end=" ")

    # Calculates average of each emotion and organizes in dictionary
    with ThreadPoolExecutor(max_workers=100) as executor:
        sentences_emotions = executor.map(get_sentence_emotions, screenplay_sentences)

    for emotion_label in emotion_labels:
            emotion_sum = sum([emotions_dict[emotion_label] for emotions_dict in sentences_emotions])
            screenplay_emotions_dict[emotion_label] = emotion_sum / sentences_count

    print("complete.")

    return screenplay_emotions_dict

def get_sentence_emotions(sentence):
    emotions_scores = sum(emotion_pipeline(sentence), []) # Flattens the list
    emotions_dict = {}

    # Organizes emotions and their scores in dictionary
    for emotion in emotions_scores:
        emotions_dict[emotion["label"].capitalize()] = emotion["score"]

    return emotions_dict

def extract_features(screenplay_title, screenplay_text):
    features_dict = {"Title": screenplay_title}

    # TODO: FURTHER FEATURE EXTRACTION (IF NECESSARY)
    features_dict.update(get_screenplay_emotions(screenplay_text))
    # features_dict.update(...)

    return features_dict

