# Imports
import nltk, re

from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Methods
def clean_text(text):
    stop_words = set(stopwords.words("english"))

    # Removes non-alphabets, stop words and excess spaces
    text = re.sub("\W+", " ", text)
    text = " ".join([word for word in text.split() if word not in stop_words])

    return text.lower()

def get_sentences(text):
    return sent_tokenize(text)

def get_words(text):
    return sum([word_tokenize(sentence) for sentence in get_sentences(text)], [])  # flattened list

def get_parts_of_speech(text):
    words = nltk.pos_tag(get_words(text), tagset="universal")
    pos_tags = set(tag for (word, tag) in words)
    pos_dict = {}

    # Organizes all parts of speech in dictionary
    for pos_tag in pos_tags:
        pos_dict[pos_tag] = set(word for (word, tag) in words if tag == pos_tag)

    return pos_dict

def lemmatize_words(words):
    lemmatizer = WordNetLemmatizer()

    return set(lemmatizer.lemmatize(word) for word in words)