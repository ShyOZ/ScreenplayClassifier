# Imports
import pandas, spacy, textblob

from nltk.tokenize import sent_tokenize

# Methods
def get_entities(nlp_info):
    entities_labels = set(entity.label_ for entity in nlp_info.ents)
    entities_dict = {}

    # Organizes entities in dictionary
    for entitiy_label in entities_labels:
        entities_dict[entitiy_label] = [entity.text for entity in nlp_info.ents if entity.label_ == entitiy_label]

    return entities_dict

def get_sentiment(text):
    sentences = sent_tokenize(text)
    sentiments_dict = {"Positive": 0, "Neutral": 0, "Negative": 0}

    # Determines text's sentiment by the largest subgroup in it: Negative, Neutral or Positive
    for sentence in sentences:
        sentiment_polarity = textblob.TextBlob(sentence).sentiment.polarity

        if sentiment_polarity < 0:
            sentiments_dict["Negative"] += 1
        elif sentiment_polarity == 0:
            sentiments_dict["Neutral"] += 1
        else:
            sentiments_dict["Positive"] += 1

    sentiments_dict = dict(sorted(sentiments_dict.items(), key=lambda item: item[1], reverse=True))

    return list(sentiments_dict.keys())[0]

def get_emotions(text):
    emotions_dict = {"Happiness": 0, "Sadness": 0, "Fear": 0, "Disgust": 0, "Anger": 0, "Contempt": 0, "Surprise": 0}

    # TODO: COMPLETE

    return emotions_dict

def process_screenplays(screenplays):
    nlp = spacy.load("en_core_web_sm")

    # Extracts features from the screenplay text
    for offset, screenplay in screenplays.iterrows():
        nlp_info = nlp(screenplay["Text"])

        screenplay["Entities"] = get_entities(nlp_info)
        screenplay["Sentiment"] = get_sentiment(screenplay["Text"])
        screenplay["Emotions"] = get_emotions(screenplay["Text"])

    # Removes the no longer required text feature
    screenplays.drop("Text", axis=1)

    return screenplays