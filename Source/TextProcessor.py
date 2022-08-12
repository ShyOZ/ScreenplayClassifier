# Imports
import nltk

from nltk.corpus import stopwords
from nltk.stem import snowball, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from re import search
from typing import List

# Methods
def get_words(text: str) -> List[str]:
    stop_words = set(stopwords.words("english"))
    text_tokens = word_tokenize(text)

    return list(filter(lambda word: search("[^a-zA-Z]", word) is None
                                    and word.lower() not in stop_words, text_tokens))

def get_sentences(text: str) -> List[str]:
    return sent_tokenize(text)

def get_tokens(text: str) -> List[str]:
    stemmer = snowball.EnglishStemmer()
    lemmatizer = WordNetLemmatizer()
    text_words = get_words(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in text_words]

    return [stemmer.stem(token) for token in lemmatized_tokens]

def get_corpus(text: str) -> dict:
    return {"Words": get_words(text), "Sentences": get_sentences(text)}
