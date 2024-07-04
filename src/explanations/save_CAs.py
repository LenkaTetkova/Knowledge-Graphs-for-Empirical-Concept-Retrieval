import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.load_data import load_data
from src.explanations.create_CAs import fit_CAR_classifier, fit_CAV_classifier
from src.explanations.utils import path_edit
from src.models.evaluate import get_latent_representations
from src.models.load_models import load_model


@hydra.main(config_path="../config", config_name="default.yaml")
def save_CAs(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    orig_cwd = hydra.utils.get_original_cwd()
    # Fix random seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(cfg.model_name, cfg.data_type, device)

    path_data = path_edit(cfg.path_data, orig_cwd)

    print(f"Model {cfg.model_name}")

    concepts = []
    for concept in ["sport", "edible_fruit", "motor_vehicle"]:
        subconcepts = np.load(f"{orig_cwd}/src/data/wikidata_ids/{concept}.npy", allow_pickle=True).item()

        for name, wiki_id in subconcepts.items():
            if wiki_id is not None and type(wiki_id) is not list and wiki_id not in concepts:
                concepts.append(wiki_id)

    for concept in concepts:
        if cfg.data_type == "text" and not os.path.exists(f"{path_data}text/extract-result_{concept}.csv"):
            print(f"Concept {concept} does not exist (text).")
            continue
        if cfg.data_type == "images" and not os.path.exists(f"{path_data}images/{concept}"):
            print(f"Concept {concept} does not exist (images).")
            continue

        X_train, X_test, y_train, y_test = load_data(
            concept,
            cfg.data_type,
            path_data,
            test_size=cfg.test_size,
            max_size_per_concept=cfg.max_size,
            seed=cfg.seed,
            concept_name_human_readable=concept,
        )
        if len(X_train) + len(X_test) < cfg.min_size:
            print(f"Concept {concept} has only {len(X_train) + len(X_test)} data.")
            continue

        for layer in cfg.layers:
            print(f"Layer {layer}")
            H_train = get_latent_representations(X_train, device, model, layer, batch_size=cfg.batch_size)
            H_test = get_latent_representations(X_test, device, model, layer, batch_size=cfg.batch_size)

            cav = fit_CAV_classifier(
                H_train,
                y_train,
                H_test,
                y_test,
                f"{path_data}CAV/{cfg.model_name}_{concept}_{layer}.pkl",
                device,
                save=True,
                batch_size=1,
            )
            car = fit_CAR_classifier(
                H_train,
                y_train,
                H_test,
                y_test,
                f"{path_data}CAR/{cfg.model_name}_{concept}_{layer}.pkl",
                device,
                save=True,
                batch_size=1,
            )
        print(f"Concept {concept} finished")


if __name__ == "__main__":
    save_CAs()
