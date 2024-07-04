import os
import random
from fnmatch import fnmatch
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from src.data.images import load_images
from src.data.text import load_text, load_wiki_texts
from src.data.utils import shuffle_and_cut


def load_data(
    concept_name: str,
    data_type: str,
    path_data: str,
    test_size: float,
    max_size_per_concept: int,
    seed: int,
    concept_name_human_readable: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads data.
    :param concept_name: Name of the concept to load.
    :param data_type: "images", "text" etc.
    :param path_data: Absolute path to the folder with data.
    :param model_name: Name of the model.
    :param test_size: Proportion of the data in the test set (between 0 and 1)
    :param max_size_per_concept: Maximum number of examples per concept.
    :return: Train and test sets of data (X_...) and labels (y_...) in np array formats.
    """
    if data_type == "images":
        if "VOC2012" in path_data:
            positive_paths, negative_paths = load_pascal_labels(path_data, concept_name)
            data_all, labels_all = shuffle_and_cut(positive_paths, negative_paths, max_size_per_concept)
        else:
            if "objectnet" in path_data or "Color" in path_data or "dtd" in path_data:
                concept_path = f"{path_data}{concept_name}/"
                random_path = path_data
            else:
                concept_path = f"{path_data}{data_type}/{concept_name}/"
                random_path = f"{path_data}{data_type}/random/"
            data_all, labels_all = get_positive_and_negative_paths(
                concept_path, random_path, max_size_per_concept, concept_name_human_readable
            )
        if test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                data_all, labels_all, test_size=test_size, random_state=seed
            )
            X_test, y_test = load_images(X_test, y_test)
        else:
            X_train = data_all
            y_train = labels_all
            X_test = []
            y_test = []
        X_train, y_train = load_images(X_train, y_train)
    elif data_type == "text":
        X_train, X_test, y_train, y_test = load_text(concept_name, path_data, test_size, max_size_per_concept, seed)
    else:
        raise NotImplementedError(f"Data type {data_type} not implemented.")
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return X_train, X_test, y_train, y_test


def get_positive_and_negative_paths(
    concept_path: str, random_path: str, max_size: int, concept_name: str
) -> Tuple[list[str], list[int]]:
    """
    Makes a list of paths to positive and negative concept examples and their respective labels.
    All positive examples need to be in the same folder (and the same for negative).
    :param concept_path: Path to a folder with positive examples of the concept.
    :param random_path: Path to a folder with negative (i.e., random) examples of the concept.
    :param max_size: Maximum number of examples per concept.
    :param concept_name: Name of the concept.
    :return: A list of paths and a list of their respective labels. Each of the lists has
    the length of at most 2*max_size. The numbers of positive and negative examples are balanced.
    """
    positive_list = []
    for item in os.listdir(concept_path):
        if os.path.isfile(os.path.join(concept_path, item)):
            positive_list.append(concept_path + item)
    negative_list = []
    pattern1 = "*.png"
    pattern2 = "*.jpg"
    for path, subdirs, files in os.walk(random_path):
        folder_name = path.split("/")[-1]
        if folder_name == concept_name:
            continue
        for name in files:
            if fnmatch(name, pattern1) or fnmatch(name, pattern2):
                negative_list.append(os.path.join(path, name))
    data_all, labels_all = shuffle_and_cut(positive_list, negative_list, max_size)
    return data_all, labels_all


def load_test_data(
    data_type: str,
    concept_name: str,
    path_data: str,
    max_size: int,
    max_length: int = 500,
) -> np.ndarray:
    if data_type == "images":
        if "VOC2012" in path_data:
            positive_list, negative_list = load_pascal_labels(path_data, concept_name)
        else:
            if "objectnet" in path_data:
                concept_path = f"{path_data}{concept_name}/"
            else:
                concept_path = f"{path_data}{data_type}/{concept_name}/"
            positive_list = []
            for item in os.listdir(concept_path):
                if os.path.isfile(os.path.join(concept_path, item)):
                    positive_list.append(concept_path + item)
        random.shuffle(positive_list)
        if len(positive_list) > max_size:
            positive_list = positive_list[:max_size]
        labels = [1] * len(positive_list)
        X_test_data, labels = load_images(positive_list, labels)
    elif data_type == "text":
        X_test_data = load_wiki_texts(f"{path_data}text/extract-result_{concept_name}.csv", concept_name)
        X_test_data = [sentence[:max_length] for sentence in X_test_data]
        random.shuffle(X_test_data)
        X_test_data = X_test_data[:max_size]
    else:
        raise NotImplementedError(f"Data type {data_type} not implemented.")
    return X_test_data


def load_sanity_data(
    concept_name: str,
    data_type: str,
    path_data: str,
    test_size: float,
    max_size_per_concept: int,
    seed: int,
    concept_name_human_readable: str,
    n_rounds: int = 10,
    max_length: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads data.
    :param concept_name: Name of the concept to load.
    :param data_type: "images", "text" etc.
    :param path_data: Absolute path to the folder with data.
    :param model_name: Name of the model.
    :param test_size: Proportion of the data in the test set (between 0 and 1)
    :param max_size_per_concept: Maximum number of examples per concept.
    :return: Train and test sets of data (X_...) and labels (y_...) in np array formats.
    """
    if data_type not in ["images", "text"]:
        raise NotImplementedError
    if data_type == "images":
        if "VOC2012" in path_data:
            positive_list, negative_list = load_pascal_labels(path_data, concept_name)
        else:
            if "objectnet" in path_data or "Color" in path_data or "dtd" in path_data:
                concept_path = f"{path_data}{concept_name}/"
                random_path = path_data
            else:
                concept_path = f"{path_data}{data_type}/{concept_name}/"
                random_path = f"{path_data}{data_type}/random/"
            positive_list = []
            for item in os.listdir(concept_path):
                if os.path.isfile(os.path.join(concept_path, item)):
                    positive_list.append(concept_path + item)
            negative_list = []
            pattern1 = "*.png"
            pattern2 = "*.jpg"
            for path, subdirs, files in os.walk(random_path):
                folder_name = path.split("/")[-1]
                if folder_name == concept_name:
                    continue
                for name in files:
                    if fnmatch(name, pattern1) or fnmatch(name, pattern2):
                        negative_list.append(os.path.join(path, name))
    else:
        positive_list = load_wiki_texts(f"{path_data}text/extract-result_{concept_name}.csv", concept_name)
        negative_list = load_wiki_texts(f"{path_data}text/extract-result_random.csv", concept_name)
        positive_list = [sentence[:max_length] for sentence in positive_list]
        negative_list = [sentence[:max_length] for sentence in negative_list]

    random.shuffle(positive_list)
    random.shuffle(negative_list)

    if len(positive_list) < max_size_per_concept:
        n_test = int(len(positive_list) * test_size)
        n_train = len(positive_list) - n_test
    else:
        n_test = int(max_size_per_concept * test_size)
        n_train = max_size_per_concept - n_test
    positive_train = positive_list[:n_train]
    positive_test = positive_list[n_train : n_train + n_test]
    data_all = []
    for i in range(n_rounds):
        negative_train = negative_list[i * n_train : (i + 1) * n_train]
        negative_test = negative_list[-(i + 2) * n_test : -(i + 1) * n_test]
        train = positive_train + negative_train
        test = positive_test + negative_test
        y_train = [1] * len(positive_train) + [0] * len(negative_train)
        y_test = [1] * len(positive_test) + [0] * len(negative_test)
        if data_type == "images":
            train, y_train = load_images(train, y_train)
            test, y_test = load_images(test, y_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        data_all.append([train, test, y_train, y_test])
    return data_all


def load_pascal_labels(path, concept, type="trainval"):
    with open(f"{path}ImageSets/Main/{concept}_{type}.txt") as file:
        lines = [line.rstrip() for line in file]
    path_images = f"{path}JPEGImages/"
    positive_list = []
    negative_list = []
    for line in lines:
        line_sp = line.split()
        if line_sp[1] == "1":
            positive_list.append(f"{path_images}{line_sp[0]}.jpg")
        else:
            negative_list.append(f"{path_images}{line_sp[0]}.jpg")
    return positive_list, negative_list
