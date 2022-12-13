# Imports
import nltk, re

from nltk.corpus import stopwords

# Methods
def clean_text(text: str):
    stop_words = set(stopwords.words("english"))

    # Removes everything except alphabets, then removes stopwords
    text = re.sub("\W+", " ", text)
    text = " ".join([word for word in text.split() if word not in stop_words])

    return text

def process_text(text: str):
    text = clean_text(text)

    # Tokenizes sentences, then tags each sentence with pos (parts of speech)
    sentences = [nltk.word_tokenize(sentence) for sentence in nltk.sent_tokenize(text)]
    sentences = sum([nltk.pos_tag(sentence) for sentence in sentences], [])

    # Performs Noun Phrase (NP) Chunking
    # e.g.: (NP the/DT little/JJ yellow/JJ dog/NN) barked/VBD at/IN (NP the/DT cat/NN)
    chunking_grammar = "NP: {<DT>?<JJ>*<NN>}"
    chunking_parser = nltk.RegexpParser(chunking_grammar)
    sentences = chunking_parser.parse(sentences)

    # Performs Named Entities Recognition
    print(nltk.ne_chunk(sentences[0]))

    return text
