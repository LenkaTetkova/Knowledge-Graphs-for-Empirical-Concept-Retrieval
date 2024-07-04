import json
import os
import random

import hydra
import numpy as np
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf

from src.data.load_data import load_pascal_labels
from src.data.text import wiki_clean_page
from src.data.utils import wiki_names_from_id
from src.explanations.utils import path_edit


@hydra.main(config_path="../config", config_name="default.yaml")
def get_overview(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    orig_cwd = hydra.utils.get_original_cwd()
    # Fix random seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    path_data = path_edit(cfg.path_data, orig_cwd)

    concepts = [
        "random",
    ]
    concepts_images = [
        "Q11442",
        "Q146",
        "Q7368",
        "Q27993793",
        "Q80228",
    ]
    concepts_text = [
        "Q1071646",  # world news, the item world is "Q16502",
        "Q650483",  # sports journalism, the item sport is  "Q349",
        "Q2142888",  # business journalism, the item business is "Q4830453",
        "Q1505283",  # science journalism, the item science is "Q336",
    ]

    print(f"Concept name \t & WikiData ID \t & Pascal VOC " f"\t & WikiMedia \t \\\\ \hline")

    for concept in concepts_images:
        human_name = wiki_names_from_id(concept)
        if concept == "Q146":
            human_name = "cat"
        if human_name == "potted plant":
            human_name = "pottedplant"
        path_concept_images = f"{path_data}images/{concept}/"
        if not os.path.exists(path_concept_images):
            n_images = 0
        else:
            n_images = len(
                [name for name in os.listdir(path_concept_images) if os.path.isfile(path_concept_images + name)]
            )
        positive_paths, negative_paths = load_pascal_labels(f"{path_data}VOC2012/", human_name)
        n_images_VOC = len(positive_paths)
        print(f"{human_name} \t & {concept} \t & {n_images_VOC} " f"\t & {n_images} \t \\\\ \hline")

    # Text
    print(
        f"Concept name \t & WikiData ID \t & AG News "
        f"\t & Wikipedia concept \t & Wikipedia subconcepts"
        f"\t & Wikipedia total \t \\\\ \hline"
    )
    label_dict = {
        "world news": 0,
        "sports journalism": 1,
        "business journalism": 2,
        "science journalism": 3,
    }
    dataset = load_dataset("ag_news", split="test")

    for concept in concepts_text:
        human_name = wiki_names_from_id(concept)
        target_label = label_dict[human_name]
        pos_indices = np.random.choice((np.array(dataset["label"]) == target_label).nonzero()[0], size=100000)

        try:
            with open(f"{path_data}text/extract-result_{concept}.csv", "r") as fid:
                lines = fid.readlines()
            n_main_article = 0
            other_articles = []
            not_found = 0
            for line in lines:
                split = line.split(",")
                id = split[0]
                rest = ",".join(split[2:])
                try:
                    rest = json.loads(rest)
                    rest = wiki_clean_page(rest["extract"])
                    rest = [sent for sent in rest if len(sent) > 50]
                    if id == concept:
                        n_main_article = len(rest)
                    else:
                        other_articles.extend(rest)
                except Exception as e:
                    not_found += 1
            n_other_articles = len(other_articles)
        except FileNotFoundError as e:
            n_main_article = 0
            n_other_articles = 0
        print(
            f"{human_name} \t & {concept} \t & {len(pos_indices)} "
            f"\t & {n_main_article} \t & {n_other_articles}"
            f"\t & {n_main_article+n_other_articles} \t \\\\ \hline"
        )

    # Other
    print(
        f"Concept name \t & WikiData ID \t & # images "
        f"\t & # sentences in main concept \t & # sentences in subconcepts"
        f"\t & # sentences in total \t \\\\ \hline"
    )
    groups = []
    group_id = 0
    for concept in ["sport", "edible_fruit", "motor_vehicle"]:
        subconcepts = np.load(f"{orig_cwd}/src/data/wikidata_ids/{concept}.npy", allow_pickle=True).item()

        for name, wiki_id in subconcepts.items():
            if wiki_id is not None and type(wiki_id) is not list and wiki_id not in concepts:
                concepts.append(wiki_id)
                groups.append(group_id)
        group_id += 1
    for concept in concepts:
        if concept == "random":
            human_name = "random"
        else:
            human_name = wiki_names_from_id(concept)
        path_concept_images = f"{path_data}images/{concept}/"
        if not os.path.exists(path_concept_images):
            n_images = 0
        else:
            n_images = len(
                [name for name in os.listdir(path_concept_images) if os.path.isfile(path_concept_images + name)]
            )
        try:
            with open(f"{path_data}text/extract-result_{concept}.csv", "r") as fid:
                lines = fid.readlines()
            n_main_article = 0
            other_articles = []
            not_found = 0
            for line in lines:
                split = line.split(",")
                id = split[0]
                rest = ",".join(split[2:])
                try:
                    rest = json.loads(rest)
                    rest = wiki_clean_page(rest["extract"])
                    rest = [sent for sent in rest if len(sent) > 50]
                    if id == concept:
                        n_main_article = len(rest)
                    else:
                        other_articles.extend(rest)
                except Exception as e:
                    not_found += 1
            n_other_articles = len(other_articles)
        except FileNotFoundError as e:
            # print(e)
            n_main_article = 0
            n_other_articles = 0
        print(
            f"{human_name} \t & {concept} \t & {n_images} "
            f"\t & {n_main_article} \t & {n_other_articles}"
            f"\t & {n_main_article+n_other_articles} \t \\\\ \hline"
        )

    n_subconcepts = [0, 0, 0]
    subconcept_names = [[], [], []]
    for i, concept in enumerate(concepts[1:]):
        path = f"{path_data}CAV/{cfg.model_name}_{concept}_0.pkl"
        if os.path.isfile(path):
            n_subconcepts[groups[i]] += 1
            human_name = wiki_names_from_id(concept)
            subconcept_names[groups[i]].append(f"'{human_name}'")
        else:
            print(path)

    for i in range(3):
        print(f"Group {i}: {n_subconcepts[i]} subconcepts.")
        print(", ".join(subconcept_names[i]))


if __name__ == "__main__":
    get_overview()
