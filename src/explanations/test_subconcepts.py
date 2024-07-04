import os
import pickle
import random
from itertools import combinations

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics.pairwise import cosine_similarity

from src.data.load_data import load_data, load_sanity_data
from src.explanations.create_CAs import fit_CAV_classifier
from src.explanations.utils import concept_to_id, path_edit, permutation_cosine
from src.models.evaluate import get_latent_representations
from src.models.load_models import load_model


def test_similarity(data, names, device, groups=None):
    cavs = []
    for concept, name in zip(data, names):
        H_train, y_train, H_test, y_test = concept
        cav = fit_CAV_classifier(H_train, y_train, H_test, y_test, "", device, save=False, batch_size=1)
        cavs.append(np.squeeze(cav.get_activation_vector().detach().cpu().numpy()))
    similarities = cosine_similarity(cavs)
    results = {}
    random_cosine = {}
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            name = f"{names[i]}_{names[j]}"
            results[name] = similarities[i][j]
            print(f"Cosine similarity between {names[i]} and {names[j]} is {similarities[i][j]}.")
            random_cosine[name] = permutation_cosine(cavs[i], cavs[j])
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
    return results, random_cosine, results_triplets


def test_sanity(data, names, device):
    cavs = []
    for concept, name in zip(data, names):
        H_train, y_train, H_test, y_test = concept
        cav = fit_CAV_classifier(H_train, y_train, H_test, y_test, "", device, save=False, batch_size=1)
        cavs.append(np.squeeze(cav.get_activation_vector().detach().cpu().numpy()))
    similarities = cosine_similarity(cavs)
    results = []
    random_cosine = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            results.append(similarities[i][j])
            print(f"Cosine similarity between {names[i]} and {names[j]} is {similarities[i][j]}.")
            random_cosine.append(permutation_cosine(cavs[i], cavs[j]))
    random_cosine = np.mean(random_cosine)
    return results, random_cosine


def test_tuples(tuple, path_data, cfg, device, model, sanity=False, groups=None):
    wiki_ids = [concept_to_id(conc) for conc in tuple]
    if sanity:
        data = load_sanity_data(
            wiki_ids[0],
            cfg.data_type,
            path_data,
            test_size=cfg.test_size,
            max_size_per_concept=cfg.max_size,
            seed=cfg.seed,
            concept_name_human_readable=tuple[0],
            n_rounds=len(tuple),
        )
    else:
        data = []
        for i in range(len(tuple)):
            X_train, X_test, y_train, y_test = load_data(
                wiki_ids[i],
                cfg.data_type,
                path_data,
                test_size=cfg.test_size,
                max_size_per_concept=cfg.max_size,
                seed=cfg.seed,
                concept_name_human_readable=tuple[i],
            )
            data.append([X_train, X_test, y_train, y_test])

    results = {}
    random_cos = {}
    results_triplets = {}
    for layer in cfg.layers:
        print(f"Layer {layer}")
        # Get latent representations
        latent_data = []
        for concept_data in data:
            X_train, X_test, y_train, y_test = concept_data
            H_train = get_latent_representations(X_train, device, model, layer, batch_size=cfg.batch_size)
            H_test = get_latent_representations(X_test, device, model, layer, batch_size=cfg.batch_size)
            latent_data.append([H_train, y_train, H_test, y_test])
        if sanity:
            results[layer], random_cos[layer] = test_sanity(latent_data, tuple, device)
        else:
            results_layer, random_layer, triplets_layer = test_similarity(latent_data, tuple, device, groups=groups)
            for key, item in results_layer.items():
                if key not in results:
                    results[key] = {}
                if key not in random_cos:
                    random_cos[key] = {}
                if key not in results_triplets:
                    results_triplets[key] = {}
                results[key][layer] = item
                random_cos[key][layer] = random_layer[key]
                results_triplets[key][layer] = triplets_layer[key]
    return results, random_cos, results_triplets


@hydra.main(config_path="../config", config_name="default.yaml")
def test_subconcepts(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    orig_cwd = hydra.utils.get_original_cwd()
    # Fix random seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(cfg.model_name, cfg.data_type, device)

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
    for concept in ["sport", "edible_fruit"]:
        subconcepts = np.load(f"{orig_cwd}/src/data/wikidata_ids/{concept}.npy", allow_pickle=True).item()

        for name, wiki_id in subconcepts.items():
            if wiki_id is not None and type(wiki_id) is not list and wiki_id not in concepts:
                concepts.append(wiki_id)
                groups.append(group_id)
        group_id += 1

    concept_tuples = [concepts]
    groups_tuples = [groups]
    for tuple, groups in zip(concept_tuples, groups_tuples):
        dict_results, cos_results, dict_triplets = test_tuples(
            tuple, path_data, cfg, device, model, groups=groups, sanity=False
        )
        for key, value in dict_results.items():
            if key not in results_tuples:
                results_tuples[key] = {}
            if key not in random_cos:
                random_cos[key] = {}
            if key not in results_triplets:
                results_triplets[key] = {}
            for layer, res in value.items():
                results_tuples[key][layer] = res
                random_cos[key][layer] = cos_results[key][layer]
                results_triplets[key][layer] = dict_triplets[key][layer]

    with open(path_tuples, "wb") as fp:
        pickle.dump(results_tuples, fp)
        pickle.dump(random_cos, fp)
        pickle.dump(results_triplets, fp)


if __name__ == "__main__":
    test_subconcepts()
