import random

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA

from src.data.load_data import load_test_data
from src.explanations.utils import concept_to_id, path_edit
from src.models.evaluate import get_latent_representations
from src.models.load_models import load_model
from src.visualization.plot_tSNE import make_legend

colors_long = [
    "#AEC7E8",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#1f77b4",
    "#FFBB78",
    "#98DF8A",
    "#FF9896",
    "#C5B0D5",
    "#C49C94",
    "#F7B6D2",
    "#C7C7C7",
    "#DBDB8D",
    "#9EDAE5",
]
colors_short = [
    "#AEC7E8",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def visualise_concepts_PCA(X, cfg, concepts, concept_names, model_name, layer, path):
    pca = PCA(n_components=2)
    X_embedded = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(cfg.fig_width, cfg.fig_width))
    # ax = plt.gca()
    if len(concepts.keys()) <= 10:
        colors = colors_short
    else:
        colors = colors_long
    for i, concept in enumerate(concepts.keys()):
        indices = concepts[concept]
        concept_name = concept_names[i]  # concept].split(",")[0]
        alpha = 0.7 if concept_name == "background" else 1
        data_to_plot = np.take(X_embedded, indices, axis=0)
        plt.scatter(
            data_to_plot[:, 0],
            data_to_plot[:, 1],
            s=6,
            marker="o",
            edgecolors="none",
            color=colors[i],
            alpha=alpha,
            clip_on=False,
        )
    plt.subplots_adjust(top=0.98, bottom=0.02, right=0.98, left=0.02, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.axis("off")

    plt.savefig(f"{path}{model_name}_{layer}_PCA.png", dpi=cfg.dpi)
    plt.close()
    if "0" in str(layer):
        nice_names = concept_names
        fig = make_legend(cfg, nice_names, colors)
        fig.savefig(f"{path}{model_name}_PCA_legend.png", bbox_inches="tight", dpi=500, pad_inches=0)
        # fig.close()
    return


@hydra.main(config_path="../config", config_name="default.yaml")
def plot_PCA(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    orig_cwd = hydra.utils.get_original_cwd()
    # Fix random seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(cfg.model_name, cfg.data_type, device)

    path_data = path_edit(cfg.path_data, orig_cwd)
    path_figures = path_edit("./reports/figures/", orig_cwd)

    concepts = [
        "sport",
        "football",
        "tennis",
        "athletics",
        "swimming",
        "gymnastics",
        "fruit",
        "banana",
        "orange",
        "apple",
        "grape",
        "cherry",
    ]
    wiki_ids = [concept_to_id(conc) for conc in concepts]
    data = []
    indices = {}
    max = 0
    for id, concept in zip(wiki_ids, concepts):
        try:
            X_test_data = load_test_data(
                cfg.data_type,
                id,
                path_data,
                cfg.max_size,
            )
            data.append(X_test_data)
            indices[concept] = list(range(max, max + len(X_test_data)))
            max = max + len(X_test_data)
        except KeyError:
            print(f"Concept {concept} not loaded.")

    for layer in cfg.layers:
        print(f"Layer {layer}")
        latent_data = []
        for concept_data in data:
            H_data = get_latent_representations(concept_data, device, model, layer, batch_size=cfg.batch_size)
            latent_data.append(H_data)
        latent_data = np.concatenate(latent_data, axis=0)
        visualise_concepts_PCA(latent_data, cfg, indices, concepts, cfg.model_name, layer, path_figures)


if __name__ == "__main__":
    plot_PCA()
