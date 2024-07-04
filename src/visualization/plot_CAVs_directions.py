import pickle
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.explanations.utils import path_edit
from src.visualization.plot_quality import plot_labeled


@hydra.main(config_path="../config", config_name="default.yaml")
def plot_CAVs_directions(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    orig_cwd = hydra.utils.get_original_cwd()
    # Fix random seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    path_outputs = path_edit(cfg.path_outputs, orig_cwd)
    path_figures = path_edit("./reports/figures/", orig_cwd)

    path_similarity = f"{path_outputs}{cfg.data_type}_{cfg.model_name}_directions.pkl"
    with open(path_similarity, "rb") as fp:
        results_similarity = pickle.load(fp)
        try:
            random_cosine = pickle.load(fp)
        except:
            random_cosine = {}

    if len(list(random_cosine.keys())) > 0:
        random_upper = []
        random_lower = []
        one_concept = list(random_cosine.keys())[0]
        for layer in random_cosine[one_concept].keys():
            means = [random_cosine[conc][layer][0] for conc in random_cosine.keys()]
            vars = [random_cosine[conc][layer][1] for conc in random_cosine.keys()]
            mean_var = np.mean(vars)
            mean_mean = np.mean(means)
            sqrt_var = np.sqrt(mean_var)
            random_upper.append(mean_mean + sqrt_var)
            random_lower.append(mean_mean - sqrt_var)

    if cfg.model_name == "roberta":
        finetuned_name = "roberta_go"
    else:
        finetuned_name = f"{cfg.model_name}_finetuned"
    path_similarity_finetuned = f"{path_outputs}{cfg.data_type}_{finetuned_name}_directions.pkl"
    with open(path_similarity_finetuned, "rb") as fp:
        results_similarity_finetuned = pickle.load(fp)
        try:
            random_cosine_finetuned = pickle.load(fp)
        except:
            random_cosine_finetuned = {}

    if len(list(random_cosine_finetuned.keys())) > 0:
        random_upper = []
        random_lower = []
        one_concept = list(random_cosine_finetuned.keys())[0]
        for layer in random_cosine_finetuned[one_concept].keys():
            means = [random_cosine_finetuned[conc][layer][0] for conc in random_cosine_finetuned.keys()]
            vars = [random_cosine_finetuned[conc][layer][1] for conc in random_cosine_finetuned.keys()]
            mean_var = np.mean(vars)
            mean_mean = np.mean(means)
            sqrt_var = np.sqrt(mean_var)
            random_upper.append(mean_mean + sqrt_var)
            random_lower.append(mean_mean - sqrt_var)

    description = "CAV: Cosine similarity"  # of generated and labeled concepts"
    file = "CAV_directions"
    data_to_plot = []
    concepts_all = []
    for concept in results_similarity.keys():
        if concept in results_similarity_finetuned.keys() and concept in [
            "bicycle",
            "bottle",
            "cat",
            "pottedplant",
            "sheep",
        ]:
            concepts_all.append(concept)
    for res_one, cos_one in zip(
        [results_similarity, results_similarity_finetuned], [random_cosine, random_cosine_finetuned]
    ):
        data_all = []
        names = []
        layers_all = []
        for concept in concepts_all:
            names.append(concept)
            layers = list(res_one[concept].keys())
            layers_all.append(layers)
            data_all.append([res_one[concept][layer] for layer in layers])

        if len(list(cos_one.keys())) > 0:
            names.append("random")
            layers_all.append(list(cos_one[one_concept].keys()))
            data_all.append(
                {
                    "upper": random_upper,
                    "lower": random_lower,
                }
            )
        data_to_plot.append(data_all)
    plot_labeled(
        data_to_plot,
        names,
        layers_all,
        y_label=description,
        titles=["Pretrained", "Fine-tuned"],
        path_save=f"{path_figures}{cfg.model_name}_{file}.png",
        lower_limit=-1,
        upper_limit=1,
    )


if __name__ == "__main__":
    plot_CAVs_directions()
