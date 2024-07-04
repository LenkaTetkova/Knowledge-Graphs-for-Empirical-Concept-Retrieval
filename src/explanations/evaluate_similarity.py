import itertools
import os
import pickle
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold

from src.data.load_data import load_data, load_test_data
from src.explanations.concept import CAR, CAV
from src.explanations.create_CAs import fit_CAR_classifier, fit_CAV_classifier
from src.explanations.utils import concept_to_id, leave_two_out_stratified, path_edit
from src.models.evaluate import get_latent_representations
from src.models.load_models import load_model


def leave_p_out(H_data, y_data, device, p=2):
    agree = 0
    total = 0
    correct_CAR = 0
    correct_CAV = 0
    for i, (train_index, test_index) in enumerate(leave_two_out_stratified(list(range(len(H_data))), groups=y_data)):
        print(f"Fold {i}:")
        print(f"  Test:  index={test_index}")
        X_train = H_data[train_index]
        y_train = y_data[train_index]
        X_test = H_data[test_index]
        y_test = y_data[test_index]
        car = fit_CAR_classifier(X_train, y_train, X_test, y_test, "", device, save=False)
        cav = fit_CAV_classifier(X_train, y_train, X_test, y_test, "", device, save=False)
        car_pred = car.predict(X_test)
        cav_pred = cav.predict(X_test)
        total += len(car_pred)
        agree += np.sum(car_pred == cav_pred)
        correct_CAR += np.sum(car_pred == y_test)
        correct_CAV += np.sum(cav_pred == y_test)
    print(f"Total agreement between CAV and CAR: {agree/total*100} %.")
    print(f"Total accuracy CAR: {correct_CAR/total*100} %.")
    print(f"Total accuracy CAV: {correct_CAV/total*100} %.")
    return {
        "cav accuracy": correct_CAV / total * 100,
        "car accuracy": correct_CAR / total * 100,
        "cav car agreement": agree / total * 100,
    }


def cross_validation(H_data, y_data, device, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits)
    agree = 0
    total = 0
    correct_CAR = 0
    correct_CAV = 0
    for i, (train_index, test_index) in enumerate(skf.split(H_data, y_data)):
        print(f"Fold {i}:")
        print(f"  Test:  index={test_index}")
        X_train = H_data[train_index]
        y_train = y_data[train_index]
        X_test = H_data[test_index]
        y_test = y_data[test_index]
        car = fit_CAR_classifier(X_train, y_train, X_test, y_test, "", device, save=False)
        cav = fit_CAV_classifier(X_train, y_train, X_test, y_test, "", device, save=False)
        car_pred = car.predict(X_test)
        cav_pred = cav.predict(X_test)
        total += len(car_pred)
        agree += np.sum(car_pred == cav_pred)
        correct_CAR += np.sum(car_pred == y_test)
        correct_CAV += np.sum(cav_pred == y_test)
    print(f"Total agreement between CAV and CAR: {agree / total * 100} %.")
    print(f"Total accuracy CAR: {correct_CAR / total * 100} %.")
    print(f"Total accuracy CAV: {correct_CAV / total * 100} %.")
    return {
        "cav accuracy": correct_CAV / total,
        "car accuracy": correct_CAR / total,
        "cav car agreement": agree / total,
    }


def evaluate(names, H_test_data, path_CAR, path_CAV, concept_name, device):
    # CAR evaluation
    car_predictions = []
    for name in names:
        # Load CAR classifier
        car = CAR(device)
        car.load(f"{path_CAR}{concept_name}{name}.pkl")
        predictions = car.predict(np.squeeze(H_test_data))
        car_predictions.append(predictions)

    for indices in itertools.combinations(range(len(names)), 2):
        (same_predictions,) = np.where(car_predictions[indices[0]] == car_predictions[indices[1]])
        same_predictions = len(same_predictions)
        print(
            f"CAR {concept_name}{names[indices[0]]} and {names[indices[1]]} agree on "
            f"{same_predictions / len(car_predictions[0]) * 100}%."
        )

    # CAV evaluation
    cav_vectors = []
    for name in names:
        # Load CAV classifier
        cav = CAV(device)
        cav.load(f"{path_CAV}{concept_name}{name}.pkl")
        activation_vector = cav.get_activation_vector()
        cav_vectors.append(activation_vector)
    for indices in itertools.combinations(range(len(names)), 2):
        sim = cosine_similarity(cav_vectors[indices[0]], cav_vectors[indices[1]])
        print(f"CAV {concept_name}{names[indices[0]]} and {names[indices[1]]} cosine similarity" f"{sim}.")
    return


@hydra.main(config_path="../config", config_name="default.yaml")
def evaluate_CAs(cfg: DictConfig) -> None:
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
    cv = True

    print(f"Model {cfg.model_name}")
    if cfg.concept_name in ["black", "blue", "brown", "green", "orange", "red", "violet", "white", "yellow"]:
        path_original = path_data + "ColorClassification/"
        wiki_id = concept_to_id(cfg.concept_name)
        names = [cfg.concept_name, wiki_id]
        paths = [path_original, path_data]
    elif cfg.concept_name in ["dotted", "striped", "zigzagged", "spiralled", "bubbled", "knitted", "chequered"]:
        path_original = path_data + "dtd/images/"
        names = [cfg.concept_name]
        paths = [path_original]
    else:
        path_original = path_data + "VOC2012/"
        wiki_id = concept_to_id(cfg.concept_name)
        names = [cfg.concept_name, wiki_id]
        paths = [path_original, path_data]

    for name, path in zip(names, paths):
        print(f"{name}")
        for train_size in cfg.train_sizes:
            print(f"{train_size} concept data.")
            X_train, X_test, y_train, y_test = load_data(
                name,
                cfg.data_type,
                path,
                test_size=cfg.test_size,
                max_size_per_concept=train_size,
                seed=cfg.seed,
                concept_name_human_readable=cfg.concept_name,
            )
            path_similarity = f"{path_outputs}{cfg.data_type}_{cfg.model_name}_similarity.pkl"
            results_new = {}
            if name == cfg.concept_name:
                concept_key = name
            else:
                concept_key = f"{cfg.concept_name}_{wiki_id}"
            concept_key = f"{concept_key}_{train_size}"

            if concept_key not in results_new:
                results_new[concept_key] = {}
            for layer in cfg.layers:
                print(f"Layer {layer}")
                # Get latent representations
                H_train = get_latent_representations(X_train, device, model, layer, batch_size=cfg.batch_size)
                H_test = get_latent_representations(X_test, device, model, layer, batch_size=cfg.batch_size)

                H_all = np.concatenate((H_train, H_test), axis=0)
                y_all = np.concatenate((y_train, y_test), axis=0)

                if cv:
                    dict_results = cross_validation(H_all, y_all, device, n_splits=cfg.n_splits)
                else:
                    dict_results = leave_p_out(H_all, y_all, device, p=2)

                results_new[concept_key][layer] = dict_results

            if os.path.isfile(path_similarity):
                with open(path_similarity, "rb") as fp:
                    results_similarity = pickle.load(fp)

            for concept in results_new.keys():
                results_similarity[concept] = results_new[concept]
            with open(path_similarity, "wb") as fp:
                pickle.dump(results_similarity, fp)


if __name__ == "__main__":
    evaluate_CAs()
