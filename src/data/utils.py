import json
import random

import requests


def shuffle_and_cut(positive_list, negative_list, max_size):
    random.shuffle(positive_list)
    random.shuffle(negative_list)

    if len(positive_list) > max_size:
        positive_list = positive_list[:max_size]
    if len(negative_list) > max_size:
        negative_list = negative_list[:max_size]
    if len(positive_list) > len(negative_list):
        positive_list = positive_list[: len(negative_list)]
    elif len(positive_list) < len(negative_list):
        negative_list = negative_list[: len(positive_list)]
    negative_labels = [0] * len(negative_list)
    positive_labels = [1] * len(positive_list)
    data_all = positive_list + negative_list
    labels_all = positive_labels + negative_labels
    return data_all, labels_all


def wiki_names_from_id(wiki_id):
    query = (
        f"https://www.wikidata.org/w/api.php?action=wbgetentities&props=labels&ids={wiki_id}&languages=en&format=json"
    )
    response = requests.get(query, headers={}, params={})
    response_json = json.loads(response.text)
    name = response_json["entities"][wiki_id]["labels"]["en"]["value"]
    return name
