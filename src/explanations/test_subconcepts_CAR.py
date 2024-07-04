import os
import pickle
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics.pairwise import cosine_similarity

from src.data.load_data import load_test_data
from src.explanations.concept import CAR, CAV
from src.explanations.create_CAs import fit_CAV_classifier
from src.explanations.utils import path_edit, permutation_cosine
from src.models.evaluate import get_latent_representations
from src.models.load_models import load_model


def test_similarity(data, names, device):
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
    return results, random_cosine


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


def test_pairs(subconcept_id, ca, path_data, cfg, device, model):
    X_subconcept = load_test_data(
        cfg.data_type,
        subconcept_id,
        path_data,
        cfg.max_size,
    )
    y_subconcept = [1] * len(X_subconcept)

    results = {}
    for layer in cfg.layers:
        print(f"Layer {layer}")
        H_subconcept = get_latent_representations(X_subconcept, device, model, layer, batch_size=cfg.batch_size)
        acc = ca[layer].evaluate(H_subconcept, y_subconcept)
        results[layer] = acc
    return results


@hydra.main(config_path="../config", config_name="default.yaml")
def test_subconcepts_CAR(cfg: DictConfig) -> None:
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

    path_pairs = f"{path_outputs}{cfg.data_type}_{cfg.model_name}_pairs.pkl"

    if os.path.isfile(path_pairs):
        with open(path_pairs, "rb") as fp:
            results_pairs = pickle.load(fp)
    else:
        results_pairs = {}
    concepts = []
    groups = []
    group_id = 0
    main_ids = []
    for concept in ["sport", "edible_fruit", "motor_vehicle"]:
        subconcepts = np.load(f"{orig_cwd}/src/data/wikidata_ids/{concept}.npy", allow_pickle=True).item()

        for name, wiki_id in subconcepts.items():
            if name == concept:
                main_ids.append(wiki_id)
            elif wiki_id is not None and type(wiki_id) is not list and wiki_id not in concepts:
                concepts.append(wiki_id)
                groups.append(group_id)
        group_id += 1
    for key in main_ids:
        cavs = {}
        cars = {}
        for layer in cfg.layers:
            try:
                cav = CAV(device, batch_size=1)
                cav.load(f"{path_data}CAV/{cfg.model_name}_{key}_{layer}.pkl")
                cavs[layer] = cav
            except Exception as e:
                print(f"CAV: concept {key}, layer {layer} not loaded.")
            try:
                car = CAR(device, batch_size=1)
                car.load(f"{path_data}CAR/{cfg.model_name}_{key}_{layer}.pkl")
                cars[layer] = car
            except Exception as e:
                print(f"CAR: concept {key}, layer {layer} not loaded.")

        for concept in concepts:
            if cfg.data_type == "text" and not os.path.exists(f"{path_data}text/extract-result_{concept}.csv"):
                print(f"Concept {concept} does not exist (text).")
                continue
            if cfg.data_type == "images" and not os.path.exists(f"{path_data}images/{concept}"):
                print(f"Concept {concept} does not exist (images).")
                continue

            name = "_".join([key, concept])
            if name not in results_pairs.keys():
                results_pairs[name] = {}
                dict_results = test_pairs(concept, cavs, path_data, cfg, device, model)
                for layer, value in dict_results.items():
                    if layer not in results_pairs[name].keys():
                        results_pairs[name][layer] = {}
                    results_pairs[name][layer]["cav"] = value

                dict_results = test_pairs(concept, cars, path_data, cfg, device, model)
                for layer, value in dict_results.items():
                    if layer not in results_pairs[name].keys():
                        results_pairs[name][layer] = {}
                    results_pairs[name][layer]["car"] = value

                with open(path_pairs, "wb") as fp:
                    pickle.dump(results_pairs, fp)


if __name__ == "__main__":
    test_subconcepts_CAR()
