import os
import pickle
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics.pairwise import cosine_similarity

from src.data.load_data import load_data
from src.explanations.create_CAs import fit_CAV_classifier
from src.explanations.utils import concept_to_id, path_edit, permutation_cosine
from src.models.evaluate import get_latent_representations
from src.models.load_models import load_model


@hydra.main(config_path="../config", config_name="default.yaml")
def evaluate_CAVs(cfg: DictConfig) -> None:
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
    path_similarity = f"{path_outputs}{cfg.data_type}_{cfg.model_name}_directions.pkl"
    if os.path.isfile(path_similarity):
        with open(path_similarity, "rb") as fp:
            results_similarity = pickle.load(fp)
            try:
                random_cosine = pickle.load(fp)
            except:
                random_cosine = {}
    else:
        results_similarity = {}
        random_cosine = {}

    wiki_id = concept_to_id(cfg.concept_name)
    if cfg.concept_name in ["black", "blue", "brown", "green", "orange", "red", "violet", "white", "yellow"]:
        path_original = path_data + "ColorClassification"
    else:
        path_original = path_data + "VOC2012/"
    cavs = {}
    for name, path in zip([cfg.concept_name, wiki_id], [path_original, path_data]):
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

        if cfg.concept_name not in results_similarity:
            results_similarity[cfg.concept_name] = {}
        if name not in cavs:
            cavs[name] = {}
        for layer in cfg.layers:
            print(f"Layer {layer}")
            # Get latent representations
            H_train = get_latent_representations(X_train, device, model, layer, batch_size=cfg.batch_size)
            H_test = get_latent_representations(X_test, device, model, layer, batch_size=cfg.batch_size)
            cav = fit_CAV_classifier(H_train, y_train, H_test, y_test, "", device, save=False)
            cavs[name][layer] = np.squeeze(cav.get_activation_vector().detach().cpu().numpy())

    if cfg.concept_name not in random_cosine:
        random_cosine[cfg.concept_name] = {}

    for layer in cfg.layers:
        vec1 = cavs[cfg.concept_name][layer]
        vec2 = cavs[wiki_id][layer]
        similarities = cosine_similarity([vec1, vec2])
        results_similarity[cfg.concept_name][layer] = similarities[0][1]
        random_cosine[cfg.concept_name][layer] = permutation_cosine(vec1, vec2)

    with open(path_similarity, "wb") as fp:
        pickle.dump(results_similarity, fp)
        pickle.dump(random_cosine, fp)


if __name__ == "__main__":
    evaluate_CAVs()
