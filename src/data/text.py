import json
import random
from typing import List, Tuple

from transformers import BatchEncoding

from src.data.utils import shuffle_and_cut


def wiki_clean_page(text):
    text_clean = text.replace("\n\n", " ")
    text_clean = text_clean.replace("\n", " ")
    text_clean = text_clean.replace("\t", "")
    text_clean = text_clean.replace("  ", " ")
    text_clean = text_clean.replace("e.g.", "e.g.,")
    phrase_to_list = text_clean.split(". ")
    return phrase_to_list


def get_concept_data_wikipedia(concept_name, path_data, max_size, test_size, max_length=500):
    positive_sentences = load_wiki_texts(f"{path_data}text/extract-result_{concept_name}.csv", concept_name)
    negative_sentences = load_wiki_texts(f"{path_data}text/extract-result_random.csv", concept_name)
    positive_sentences = [sentence[:max_length] for sentence in positive_sentences]
    negative_sentences = [sentence[:max_length] for sentence in negative_sentences]
    data_all, labels_all = shuffle_and_cut(positive_sentences, negative_sentences, max_size)
    n_test = int(len(data_all) * test_size)
    X_test = data_all[:n_test]
    X_train = data_all[n_test:]
    y_test = labels_all[:n_test]
    y_train = labels_all[n_test:]
    return X_train, X_test, y_train, y_test


def load_wiki_texts(file, concept, min_length=50, n_sentences=-1):
    with open(file, "r") as fid:
        lines = fid.readlines()
    main_article = []
    other_articles = []
    not_loaded = 0
    for line in lines:
        split = line.split(",")
        id = split[0]
        rest = ",".join(split[2:])
        try:
            rest = json.loads(rest)
            rest = wiki_clean_page(rest["extract"])
            rest = [sent for sent in rest if len(sent) > min_length]
            if id == concept:
                main_article = rest
            else:
                other_articles.extend(rest)
        except Exception as e:
            not_loaded += 1
    print(f"Concept {concept}: not loaded: {not_loaded}.")
    random.shuffle(main_article)
    random.shuffle(other_articles)
    sentences = main_article[:n_sentences]
    if n_sentences == -1 or len(sentences) < n_sentences:
        difference = n_sentences - len(sentences)
        sentences.extend(other_articles[:difference])
    return sentences


def load_text(
    concept_name: str,
    path_data: str,
    test_size: float,
    max_size_per_concept: int,
    seed: int,
) -> Tuple[BatchEncoding, BatchEncoding, List[int], List[int]]:
    """
    Load text and transform them according to the model to be used.
    :param concept_name: Name of the concept to load.
    :param model_name: Model name. Important for loading the correct
    transformation of the input data.
    :param test_size: Proportion of the data in the test set (between 0 and 1)
    :param max_size_per_concept: Maximum number of examples per concept.

    :return: Array of images after transformation.
    """
    X_train, X_test, y_train, y_test = get_concept_data_wikipedia(
        concept_name, path_data, max_size_per_concept, test_size
    )
    return X_train, X_test, y_train, y_test
