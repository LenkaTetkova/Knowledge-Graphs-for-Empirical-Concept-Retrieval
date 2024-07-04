import copy

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def path_edit(path: str, orig_cwd: str) -> str:
    """
    Decides whether path is absolute or relative. If relative, makes it absolute
    by appending it after the current working directory.
    :param path: Absolute or relative path. If it starts with ".", it is considered relative.
    :param orig_cwd: Current working directory
    :return: Absolute path
    """
    if path[0] == ".":
        return orig_cwd + path[1:]
    else:
        return path


def leave_two_out_stratified(indices, groups):
    group_1 = np.where(groups == 1)[0]
    group_2 = np.where(groups == 0)[0]
    print(f"Group 1: {group_1}")
    print(f"Group 2: {group_2}")
    for i in range(len(group_1)):
        for j in range(len(group_2)):
            train = copy.deepcopy(indices)
            train.remove(indices[group_1[i]])
            train.remove(indices[group_2[j]])
            yield train, [indices[group_1[i]], indices[group_2[j]]]


def concept_to_id(concept_name):
    id_dict = {
        # For ObjectNet
        "skirt": "Q2160801",
        "banana": "Q503",
        "basket": "Q201097",
        "dish_soap": "Q1517816",
        "newspaper": "Q11032",
        "wheel": "Q446",
        # For AG News
        "world": "Q1071646",  # world news, the item world is "Q16502",
        "sports": "Q650483",  # sports journalism, the item sport is  "Q349",
        "business": "Q2142888",  # business journalism, the item business is "Q4830453",
        "science": "Q1505283",  # science journalism, the item science is "Q336",
        # For subconcepts
        "sport": "Q349",
        "football": "Q2736",
        "tennis": "Q847",
        "athletics": "Q542",
        "ice sports": "Q31883501",
        "swimming": "Q31920",
        "gymnastics": "Q43450",
        "fruit": "Q3314483",
        "orange_fruit": "Q13191",
        "apple": "Q89",
        "strawberry": "Q14458220",
        "grape": "Q10978",
        "cherry": "Q196",
        "clothes": "Q11460",
        "t-shirt": "Q131151",
        "jacket": "Q849964",
        "trousers": "Q39908",
        "hat": "Q80151",
        "dress": "Q200539",
        "motor_vehicle": "Q1420",
        "acrobatics": "Q193036",
        "berry": "Q13184",
        "blackberry": "Q19842373",
        "tractor": "Q39495",
        "truck": "Q43193",
        # Colors
        "black": "Q23445",
        "blue": "Q1088",
        "brown": "Q47071",
        "green": "Q3133",
        "orange": "Q39338",
        "red": "Q3142",
        "violet": "Q428124",
        "white": "Q23444",
        "yellow": "Q943",
        # Pascal VOC
        "bicycle": "Q11442",
        "cat": "Q146",
        "sheep": "Q7368",
        "pottedplant": "Q27993793",
        "bottle": "Q80228",
    }
    return id_dict[concept_name]


def permutation_cosine(vec1, vec2, n_repetitions=100):
    similarities = []
    for i in range(n_repetitions):
        permuted_vec1 = np.random.permutation(vec1.copy())
        similarities.append(cosine_similarity([permuted_vec1, vec2])[0][1])
    mean = np.mean(similarities)
    variance = np.var(similarities)
    return (mean, variance)
