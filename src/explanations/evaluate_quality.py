import os
import pickle
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.load_data import load_data
from src.explanations.create_CAs import fit_CAR_classifier, fit_CAV_classifier
from src.explanations.utils import concept_to_id, path_edit
from src.models.evaluate import get_latent_representations
from src.models.load_models import load_model


def test(H_train, y_train, H_test, y_test, H_labeled, y_labeled, device, names):
    car = fit_CAR_classifier(H_train, y_train, H_test, y_test, "", device, save=False, batch_size=1)
    cav = fit_CAV_classifier(H_train, y_train, H_test, y_test, "", device, save=False, batch_size=1)
    acc_car = car.evaluate(H_labeled, y_labeled)
    print(f"Trained on {names[0]}, evaluating on {names[1]}.")
    print("Accuracy CAR: ", acc_car)
    acc_cav = cav.evaluate(H_labeled, y_labeled)
    print("Accuracy CAV: ", acc_cav)
    return {"cav": acc_cav, "car": acc_car}


@hydra.main(config_path="../config", config_name="default.yaml")
def evaluate_quality(cfg: DictConfig) -> None:
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
    wiki_id = concept_to_id(cfg.concept_name)

    X_train, X_test, y_train, y_test = load_data(
        wiki_id,
        cfg.data_type,
        path_data,
        test_size=cfg.test_size,
        max_size_per_concept=cfg.max_size,
        seed=cfg.seed,
        concept_name_human_readable=cfg.concept_name,
    )

    if cfg.concept_name in ["black", "blue", "brown", "green", "orange", "red", "violet", "white", "yellow"]:
        path_original = path_data + "ColorClassification"
    else:
        path_original = path_data + "VOC2012/"

    X_labeled, _, y_labeled, _ = load_data(
        cfg.concept_name,
        cfg.data_type,
        path_original,
        test_size=0,
        max_size_per_concept=cfg.max_size,
        seed=cfg.seed,
        concept_name_human_readable=cfg.concept_name,
    )

    path_labeled = f"{path_outputs}{cfg.data_type}_{cfg.model_name}_labeled.pkl"
    if os.path.isfile(path_labeled):
        with open(path_labeled, "rb") as fp:
            results_labeled = pickle.load(fp)
    else:
        results_labeled = {}

    if cfg.concept_name not in results_labeled.keys():
        results_labeled[cfg.concept_name] = {}
    for layer in cfg.layers:
        print(f"Layer {layer}")
        # Get latent representations
        H_train = get_latent_representations(X_train, device, model, layer, batch_size=cfg.batch_size)
        H_test = get_latent_representations(X_test, device, model, layer, batch_size=cfg.batch_size)
        H_labeled = get_latent_representations(X_labeled, device, model, layer, batch_size=cfg.batch_size)

        dict_results = test(
            H_train,
            y_train,
            H_test,
            y_test,
            H_labeled,
            y_labeled,
            device,
            (cfg.concept_name, cfg.concept_name + " labeled"),
        )
        results_labeled[cfg.concept_name][layer] = dict_results

        with open(path_labeled, "wb") as fp:
            pickle.dump(results_labeled, fp)


if __name__ == "__main__":
    evaluate_quality()
