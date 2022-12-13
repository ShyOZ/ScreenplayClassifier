# Imports
import nltk, re

# Methods
def clean_text(text: str):
    stop_words = set(nltk.corpus.stopwords.words("english"))

    # Removes non-alphabets, stop words and excess spaces
    text = re.sub("\W+", " ", text)
    text = " ".join([word for word in text.split() if word not in stop_words])

    return text

def process_text(text: str):
    text = clean_text(text)
    #
    # # Tags parts of speech in text sentences
    # sentences = [nltk.word_tokenize(sentence) for sentence in nltk.sent_tokenize(text)]
    # sentences = sum([nltk.pos_tag(sentence) for sentence in sentences], [])

    return text
