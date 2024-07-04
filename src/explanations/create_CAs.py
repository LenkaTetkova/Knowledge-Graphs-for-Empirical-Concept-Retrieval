import os
import pickle
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.load_data import load_data
from src.explanations.concept import CAR, CAV
from src.explanations.utils import concept_to_id, path_edit
from src.models.evaluate import get_latent_representations
from src.models.load_models import load_model


def fit_CAR_classifier(H_train, y_train, H_test, y_test, path_to_save, device, save=True, batch_size=1):
    car = CAR(device, batch_size=batch_size)
    car.fit(H_train, y_train)
    car.tune_kernel_width(H_train, y_train)
    if save:
        car.save(path_to_save)
    acc = car.evaluate(H_test, y_test)
    print(f"CAR Test accuracy: {acc}")
    return car


def fit_CAV_classifier(H_train, y_train, H_test, y_test, path_to_save, device, save=True, batch_size=1):
    cav = CAV(device, batch_size=batch_size)
    cav.fit(H_train, y_train)
    if save:
        cav.save(path_to_save)
    acc = cav.evaluate(H_test, y_test)
    print(f"CAV Test accuracy: {acc}")
    return cav


@hydra.main(config_path="../config", config_name="default.yaml")
def create_CAs(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    orig_cwd = hydra.utils.get_original_cwd()
    # Fix random seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(cfg.model_name, cfg.data_type, device)

    path_data = path_edit(cfg.path_data, orig_cwd)
    path_CAs = path_edit(cfg.path_CAs, orig_cwd)
    path_CAV = f"{path_CAs}/CAV/"
    path_CAR = f"{path_CAs}/CAR/"
    for path in [path_CAV, path_CAR]:
        if not os.path.exists(path):
            os.makedirs(path)

    print(f"Model {cfg.model_name}")
    wiki_id = concept_to_id(cfg.concept_name)
    for name, path in zip([cfg.concept_name, wiki_id], [path_data + "VOC2012/", path_data]):
        print(f"{name}")
        X_train, X_test, y_train, y_test = load_data(
            name,
            cfg.data_type,
            path,
            test_size=cfg.test_size,
            max_size_per_concept=cfg.max_size,
            seed=cfg.seed,
            concept_name_human_readable=cfg.concept_name,
        )

        for layer in cfg.layers:
            print(f"Layer {layer}")
            # Get latent representations
            H_train = get_latent_representations(X_train, device, model, layer, batch_size=cfg.batch_size)
            H_test = get_latent_representations(X_test, device, model, layer, batch_size=cfg.batch_size)

            if "objectnet" in path:
                add_name = "_objectnet"
            else:
                add_name = ""

            # Fit CAR classifier
            _ = fit_CAR_classifier(H_train, y_train, H_test, y_test, f"{path_CAR}{name}{add_name}_{layer}.pkl", device)

            # Fit CAV classifier
            _ = fit_CAV_classifier(H_train, y_train, H_test, y_test, f"{path_CAV}{name}{add_name}_{layer}.pkl", device)


if __name__ == "__main__":
    create_CAs()
