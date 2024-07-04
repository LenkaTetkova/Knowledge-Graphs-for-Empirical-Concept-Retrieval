import random

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.load_data import load_data, load_test_data
from src.explanations.utils import concept_to_id, path_edit


def plot_images(images, rows, columns, path):
    fig, ax = plt.subplots(rows, columns, sharex=True, sharey=True, layout="compressed")
    for i in range(rows):
        for j in range(columns):
            ax[i][j].imshow(images[i * columns + j])
            ax[i][j].axis("off")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return


@hydra.main(config_path="../config", config_name="default.yaml")
def plot_data(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    orig_cwd = hydra.utils.get_original_cwd()
    # Fix random seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    path_data = path_edit(cfg.path_data, orig_cwd)
    path_figures = path_edit("./reports/figures/", orig_cwd)

    concepts = ["bicycle", "bottle", "cat", "pottedplant", "sheep"]
    rows = 3
    columns = 2

    for concept_name in concepts:
        wiki_id = concept_to_id(concept_name)

        X_data = load_test_data(
            "images",
            concept_name,
            path_data + "VOC2012/",
            rows * columns,
        )
        plot_images(X_data, rows, columns, f"{path_figures}{concept_name}.png")

        X_wiki = load_test_data(
            "images",
            wiki_id,
            path_data,
            rows * columns,
        )
        plot_images(X_wiki, rows, columns, f"{path_figures}{concept_name}_{wiki_id}.png")


if __name__ == "__main__":
    plot_data()
