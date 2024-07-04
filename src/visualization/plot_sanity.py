import pickle
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.stats import sem

from src.explanations.utils import path_edit
from src.visualization.plot_similarity import plot_lines


@hydra.main(config_path="../config", config_name="default.yaml")
def plot_sanity(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    orig_cwd = hydra.utils.get_original_cwd()
    # Fix random seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    path_outputs = path_edit(cfg.path_outputs, orig_cwd)
    path_figures = path_edit("./reports/figures/", orig_cwd)

    path_sanity = f"{path_outputs}{cfg.data_type}_{cfg.model_name}_sanity.pkl"
    with open(path_sanity, "rb") as fp:
        results_sanity = pickle.load(fp)

    description = "Cosine similarity of CAVs"
    file = "CAV_sanity"
    data_all = []
    names = []
    layers_all = []
    means = {}
    stds = {}
    for concept in results_sanity.keys():
        names.append(concept)
        layers = list(results_sanity[concept].keys())
        layers_all.append(layers)
        means_layers = np.mean([results_sanity[concept][layer] for layer in layers])
        stds_layers = sem([results_sanity[concept][layer] for layer in layers])
        means[concept] = means_layers
        stds[concept] = stds_layers
        data_all.append([np.mean(results_sanity[concept][layer]) for layer in layers])

    plot_lines(
        data_all,
        names,
        layers_all,
        y_label=description,
        title=f"{description} - {cfg.model_name}",
        path_save=f"{path_figures}/{cfg.model_name}_{file}.png",
        upper_limit=1,
        lower_limit=-1,
    )
    for concept in results_sanity.keys():
        print(f"{concept} & {means[concept]: .3f}" f"$\\pm$ {stds[concept]:.3f} \t ")
    print(
        f"{cfg.model_name} \t & {means['sport']:.2f} "
        f"$\\pm$ {stds['sport']:.2f} \t "
        f"& {means['fruit']: .2f}"
        f"$\\pm$ {stds['fruit']:.2f} \t "
        f"& {means['motor_vehicle']: .2f}"
        f"$\\pm$ {stds['motor_vehicle']:.2f} \t "
    )
    print(
        f"In total & {np.mean([means[concept] for concept in results_sanity.keys()]): .3f}"
        f"$\\pm$ {np.mean([stds[concept] for concept in results_sanity.keys()]):.3f} \t \\\\ \\hline"
    )


if __name__ == "__main__":
    plot_sanity()
