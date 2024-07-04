import os
import pickle
import random
from itertools import combinations

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics.pairwise import cosine_similarity

from src.explanations.concept import CAV
from src.explanations.utils import path_edit, permutation_cosine


def test_triplets(similarities, groups, names):
    results_triplets = {}
    best_pairs = np.zeros((len(names), len(names)), dtype=int)
    tried_pairs = np.zeros((len(names), len(names)), dtype=int)
    if groups is not None:
        for i, j, k in combinations(list(range(len(names))), 3):
            same_groups = np.sum([groups[i] == groups[j], groups[i] == groups[k], groups[j] == groups[k]])
            if same_groups == 1:
                ij = similarities[i][j]
                ik = similarities[i][k]
                jk = similarities[j][k]
                assigned = False
                if ij > ik and ij > jk:
                    best_pairs[i][j] += 1
                    best_pairs[j][i] += 1
                    assigned = True
                elif ik > ij and ik > jk:
                    best_pairs[i][k] += 1
                    best_pairs[k][i] += 1
                    assigned = True
                elif jk > ij and jk > ik:
                    best_pairs[j][k] += 1
                    best_pairs[k][j] += 1
                    assigned = True
                if assigned:
                    tried_pairs[i][j] += 1
                    tried_pairs[j][i] += 1
                    tried_pairs[i][k] += 1
                    tried_pairs[k][i] += 1
                    tried_pairs[j][k] += 1
                    tried_pairs[k][j] += 1
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if tried_pairs[i, j] > 0:
                    name = f"{names[i]}_{names[j]}"
                    results_triplets[name] = best_pairs[i][j] / tried_pairs[i][j]
    return results_triplets


def test_averages(similarities, cavs, names):
    results = {}
    random_cosine = {}
    for i in range(len(cavs)):
        for j in range(i + 1, len(cavs)):
            name = f"{names[i]}_{names[j]}"
            results[name] = similarities[i][j]
            # print(f"Cosine similarity between {names[i]} and {names[j]} is {similarities[i][j]}.")
            random_cosine[name] = permutation_cosine(cavs[i], cavs[j])
    return results, random_cosine


def compute_similarities(cavs, names, groups=None):
    similarities = cosine_similarity(cavs)
    results, random_cosine = test_averages(similarities, cavs, names)
    results_triplets = test_triplets(similarities, groups, names)
    return results, random_cosine, results_triplets


@hydra.main(config_path="../config", config_name="default.yaml")
def test_subconcepts_from_saved(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    orig_cwd = hydra.utils.get_original_cwd()
    # Fix random seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    path_data = path_edit(cfg.path_data, orig_cwd)
    path_outputs = path_edit(cfg.path_outputs, orig_cwd)

    print(f"Model {cfg.model_name}")

    path_tuples = f"{path_outputs}{cfg.data_type}_{cfg.model_name}_tuples.pkl"
    if os.path.isfile(path_tuples):
        with open(path_tuples, "rb") as fp:
            results_tuples = pickle.load(fp)
            try:
                random_cos = pickle.load(fp)
            except:
                random_cos = {}
            try:
                results_triplets = pickle.load(fp)
            except:
                results_triplets = {}
    else:
        results_tuples = {}
        random_cos = {}
        results_triplets = {}
    concepts = []
    groups = []
    group_id = 0
    for concept in ["sport", "edible_fruit", "motor_vehicle"]:
        subconcepts = np.load(f"{orig_cwd}/src/data/wikidata_ids/{concept}.npy", allow_pickle=True).item()

        for name, wiki_id in subconcepts.items():
            if wiki_id is not None and type(wiki_id) is not list and wiki_id not in concepts:
                concepts.append(wiki_id)
                groups.append(group_id)
        group_id += 1

    concept_tuples = [concepts]
    groups_tuples = [groups]
    for tuple, groups in zip(concept_tuples, groups_tuples):
        for layer in cfg.layers:
            print(f"Layer {layer}")
            groups_existing = []
            names = []
            cavs = []
            for i, concept in enumerate(tuple):
                try:
                    cav = CAV(device, batch_size=1)
                    cav.load(f"{path_data}CAV/{cfg.model_name}_{concept}_{layer}.pkl")
                    cavs.append(np.squeeze(cav.get_activation_vector().detach().cpu().numpy()))
                    groups_existing.append(groups[i])
                    names.append(concept)
                except Exception as e:
                    print(f"Concept {concept} not loaded.")
            print(
                f"Number of subconcepts: {len([1 for gr in groups_existing if gr == 0])} sports, "
                f"{len([1 for gr in groups_existing if gr == 1])} edible fruit and "
                f"{len([1 for gr in groups_existing if gr == 2])} motor vehicles."
            )
            dict_results, cos_results, dict_triplets = compute_similarities(cavs, names, groups_existing)
            for key, value in dict_results.items():
                if key not in results_tuples:
                    results_tuples[key] = {}
                if key not in random_cos:
                    random_cos[key] = {}
                if key not in results_triplets:
                    results_triplets[key] = {}
                results_tuples[key][layer] = value
                random_cos[key][layer] = cos_results[key]
                results_triplets[key][layer] = dict_triplets[key]

    with open(path_tuples, "wb") as fp:
        pickle.dump(results_tuples, fp)
        pickle.dump(random_cos, fp)
        pickle.dump(results_triplets, fp)


if __name__ == "__main__":
    test_subconcepts_from_saved()
