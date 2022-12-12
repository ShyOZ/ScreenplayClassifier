# Imports

# from re import sub
# from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# Methods
def tokenize_sentences(text: str):
    # stop_words = set(stopwords.words("english"))
    # text = re.sub("\W+", " ", text)                                             # Removes everything except alphabets
    # text = " ".join([word for word in text.split() if word not in stop_words])  # Removes stopwords + spaces
    return sent_tokenize(text)