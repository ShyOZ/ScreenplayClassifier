# Imports
import pandas, spacy

# Methods
def build_entities_dict(nlp_info):
    entities_labels = set(entity.label_ for entity in nlp_info.ents)
    entities_dict = {}

    # Organizes entities in dictionary
    for entitiy_label in entities_labels:
        entities_dict[entitiy_label] = [entity.text for entity in nlp_info.ents if entity.label_ == entitiy_label]

    return entities_dict

def build_sentiments_dict(nlp_info):
    # TODO: COMPLETE
    pass


def process_screenplays(screenplays):
    nlp = spacy.load("en_core_web_sm")

    # Extracts features from the screenplay text
    for offset, screenplay in screenplays.iterrows():
        nlp_info = nlp(screenplay["Text"])

        screenplay["Entities"] = build_entities_dict(nlp_info)
        screenplay["Sentiments"] = build_sentiments_dict(nlp_info)

    # Removes the no longer required text feature
    screenplays.drop("Text", axis=1)

    return screenplays