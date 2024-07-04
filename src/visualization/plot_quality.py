import pickle
import random

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.explanations.utils import path_edit

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


def plot_labeled(
    data_all,
    names,
    layers,
    y_label,
    titles,
    path_save,
    lower_limit=0,
    upper_limit=105,
):
    fs = 10  # font size
    rcParams = {
        "text.usetex": False,
        "font.size": fs,
        "axes.labelsize": fs,
        "axes.titlesize": fs,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "legend.fontsize": fs,
        "axes.axisbelow": True,
    }
    fig, axs = plt.subplots(1, len(data_all), sharey=True, figsize=(9, 3.5))
    for i, data_one in enumerate(data_all):
        for data, name, layer in zip(data_one, names, layers):
            if name != "random":
                axs[i].plot(
                    layer,
                    data,
                    f"o-",
                    label=name,
                )
            else:
                if type(data) is dict:
                    axs[i].fill_between(
                        layer,
                        data["lower"],
                        data["upper"],
                        alpha=0.2,
                    )
                else:
                    axs[i].fill_between(
                        layer,
                        [-1 * val for val in data],
                        data,
                        alpha=0.2,
                    )
        axs[i].set_title(titles[i])
        axs[i].set_xticks(list(range(13)))
        if upper_limit > 2:
            axs[i].set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        axs[i].grid(True, which="both")
        axs[i].set_xlabel("Layers")
        axs[i].set_ylim([lower_limit, upper_limit])
    plt.rcParams.update(rcParams)
    plt.ylabel(y_label)
    if len(names) < 7:
        lgd = plt.legend(ncol=1, fontsize=fs)
    else:
        lgd = plt.legend(loc="lower right", ncols=3, bbox_to_anchor=(0, 0.0, 1, 1), fontsize=10)

    plt.savefig(path_save, dpi=600, bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.close()
    return


@hydra.main(config_path="../config", config_name="default.yaml")
def plot_quality(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    orig_cwd = hydra.utils.get_original_cwd()
    # Fix random seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    path_outputs = path_edit(cfg.path_outputs, orig_cwd)
    path_figures = path_edit("./reports/figures/", orig_cwd)

    path_similarity = f"{path_outputs}{cfg.data_type}_{cfg.model_name}_labeled.pkl"
    with open(path_similarity, "rb") as fp:
        results_similarity = pickle.load(fp)

    if cfg.model_name == "roberta":
        finetuned_name = "roberta_go"
    else:
        finetuned_name = f"{cfg.model_name}_finetuned"
    path_similarity_finetuned = f"{path_outputs}{cfg.data_type}_{finetuned_name}_labeled.pkl"
    with open(path_similarity_finetuned, "rb") as fp:
        results_similarity_finetuned = pickle.load(fp)

    for key, description, file in zip(
        ["cav", "car"],
        ["Accuracy of CAVs on labeled data in %", "Accuracy of CARs on labeled data in %"],
        ["CAV_labeled", "CAR_labeled"],
    ):
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
        for res_one in [results_similarity, results_similarity_finetuned]:
            data_all = []
            names = []
            layers_all = []
            for concept in concepts_all:
                names.append(concept)
                layers = list(res_one[concept].keys())
                layers = [l for l in layers if type(l) == int]
                layers_all.append(layers)
                results = [res_one[concept][layer][key] for layer in layers]
                if np.all([res <= 1 for res in results]):
                    results = [res * 100 for res in results]
                data_all.append(results)
            data_to_plot.append(data_all)
        plot_labeled(
            data_to_plot,
            names,
            layers_all,
            y_label=description,
            titles=["Pretrained", "Fine-tuned"],
            path_save=f"{path_figures}{cfg.model_name}_{file}.png",
        )


if __name__ == "__main__":
    plot_quality()
