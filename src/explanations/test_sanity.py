import os
import pickle
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.explanations.test_subconcepts import test_tuples
from src.explanations.utils import path_edit
from src.models.load_models import load_model


@hydra.main(config_path="../config", config_name="default.yaml")
def test_sanity(cfg: DictConfig) -> None:
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

    path_sanity = f"{path_outputs}{cfg.data_type}_{cfg.model_name}_sanity.pkl"
    if os.path.isfile(path_sanity):
        with open(path_sanity, "rb") as fp:
            results_sanity = pickle.load(fp)
            try:
                random_cos = pickle.load(fp)
            except:
                random_cos = {}
    else:
        results_sanity = {}
        random_cos = {}

    concept_tuples = [[cfg.concept_name] * 10]
    results_sanity[cfg.concept_name] = {}
    for tuple in concept_tuples:
        dict_results, cos_results, _ = test_tuples(
            tuple,
            path_data,
            cfg,
            device,
            model,
            sanity=True,
        )
        for layer, value in dict_results.items():
            results_sanity[cfg.concept_name][layer] = value
            random_cos[layer] = cos_results[layer]

    with open(path_sanity, "wb") as fp:
        pickle.dump(results_sanity, fp)
        pickle.dump(random_cos, fp)


if __name__ == "__main__":
    test_sanity()
