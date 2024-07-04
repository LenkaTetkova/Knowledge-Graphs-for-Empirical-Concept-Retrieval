import pickle
import random

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.explanations.utils import path_edit
from src.visualization.plot_quality import plot_labeled

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


def plot_lines(
    data_all,
    names,
    layers,
    y_label,
    title,
    path_save,
    lower_limit=0,
    upper_limit=105,
):
    fs = 14  # font size
    rcParams = {
        "text.usetex": False,
        "font.size": fs,
        "axes.labelsize": fs,
        "axes.titlesize": fs,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "legend.fontsize": fs,
        "axes.axisbelow": True,  # draw gridlines below other elements
        "axes.prop_cycle": plt.cycler(color=plt.get_cmap("tab20").colors),
    }
    plt.subplots(figsize=(6, 4.5))
    for data, name, layer in zip(data_all, names, layers):
        if name != "random":
            plt.plot(
                layer,
                data,
                f"o-",
                label=name,
            )
        else:
            if type(data) is dict:
                plt.fill_between(
                    layer,
                    data["lower"],
                    data["upper"],
                    alpha=0.2,
                )
            else:
                plt.fill_between(
                    layer,
                    [-1 * val for val in data],
                    data,
                    alpha=0.2,
                )
    plt.rcParams.update(rcParams)
    plt.xticks(list(range(13)))
    if upper_limit > 2:
        plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.grid(True, which="both")
    lgd = plt.legend(ncol=1, fontsize=10, bbox_to_anchor=(1.05, 0.5), loc="center left")
    plt.xlabel("Layers")
    plt.ylabel(y_label)
    plt.ylim([lower_limit, upper_limit])
    plt.title(title, fontsize=12)
    plt.savefig(path_save, dpi=600, bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.close()
    return


def plot_sizes(
    data_all,
    names,
    layers,
    y_label,
    title,
    path_save,
    lower_limit=0,
    upper_limit=100,
):
    fs = 14  # font size
    rcParams = {
        "text.usetex": False,
        "font.size": fs,
        "axes.labelsize": fs,
        "axes.titlesize": fs,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "legend.fontsize": fs,
        "axes.axisbelow": True,  # draw gridlines below other elements
    }
    plt.subplots(figsize=(6, 4.5))
    symbols = ["o", "v", "s", "*", "D"]
    colors = plt.get_cmap("tab10").colors
    for data, name, layer in zip(data_all, names, layers):
        if "(50)" in name:
            continue
        elif "(100)" in name:
            col = colors[1]
        elif "(200)" in name:
            continue
        elif "(500)" in name:
            continue
        elif "(1000)" in name:
            col = colors[4]
        elif "(30)" in name:
            col = colors[8]

        if "bicycle" in name:
            symb = symbols[0]
        elif "bottle" in name:
            symb = symbols[1]
        elif "cat" in name:
            symb = symbols[2]
        elif "pottedplant" in name:
            symb = symbols[3]
        elif "sheep" in name:
            symb = symbols[4]

        plt.plot(
            layer,
            data,
            f"{symb}-",
            label=name,
            color=col,
            lw=1,
        )
        plt.rcParams.update(rcParams)
    plt.xticks(list(range(13)))
    if upper_limit > 2:
        plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.grid(True, which="both")
    lgd = plt.legend(ncol=1, fontsize=10, bbox_to_anchor=(1.05, 0.5), loc="center left")
    plt.xlabel("Layers")
    plt.ylabel(y_label)
    plt.ylim([lower_limit, upper_limit])
    plt.title(title, fontsize=12)
    plt.savefig(path_save, dpi=600, bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.close()
    return


def plot_sizes_pairs(
    data_all,
    names,
    layers,
    y_label,
    titles,
    path_save,
    lower_limit=0,
    upper_limit=100,
):
    fs = 14  # font size
    rcParams = {
        "text.usetex": False,
        "font.size": fs,
        "axes.labelsize": fs,
        "axes.titlesize": fs,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "legend.fontsize": fs,
        "axes.axisbelow": True,  # draw gridlines below other elements
        "axes.prop_cycle": plt.cycler(color=colors_long),
    }
    symbols = ["o", "v", "s", "*", "D"]
    colors = plt.get_cmap("tab10").colors
    #
    fig, axs = plt.subplots(1, len(data_all), sharey=True, figsize=(9, 4.5))
    for i, data_one in enumerate(data_all):
        for data, name, layer in zip(data_one, names, layers):
            if "(50)" in name:
                col = colors[0]
            elif "(100)" in name:
                continue
            elif "(200)" in name:
                col = colors[1]
            elif "(500)" in name:
                continue
            elif "(1000)" in name:
                col = colors[4]
            elif "(30)" in name:
                col = colors[8]

            if "bicycle" in name:
                symb = symbols[0]
            elif "bottle" in name:
                symb = symbols[1]
            elif "cat" in name:
                symb = symbols[2]
            elif "pottedplant" in name:
                symb = symbols[3]
            elif "sheep" in name:
                symb = symbols[4]

            if name == "banana (1000)":
                name = "banana (462)"
            if name == "basket (1000)":
                name = "basket (722)"
            axs[i].plot(
                layer,
                data,
                f"{symb}-",
                label=name,
                color=col,
                lw=1,
                markersize=4,
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
    if len(names) < 6:
        lgd = plt.legend(ncol=1)
    else:
        lgd = plt.legend(loc="lower right", ncols=4, fontsize=9, bbox_to_anchor=(0, 0, 1, 1))

    plt.savefig(path_save, dpi=600, bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.close()
    return


@hydra.main(config_path="../config", config_name="default.yaml")
def plot_similarity(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    orig_cwd = hydra.utils.get_original_cwd()
    # Fix random seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    path_outputs = path_edit(cfg.path_outputs, orig_cwd)
    path_figures = path_edit("./reports/figures/", orig_cwd)

    path_similarity = f"{path_outputs}{cfg.data_type}_{cfg.model_name}_similarity.pkl"
    with open(path_similarity, "rb") as fp:
        results_similarity = pickle.load(fp)

    concept_names = []
    color_names = []
    texture_names = []
    for concept in results_similarity.keys():
        orig_name = concept.split("_")[0]
        if orig_name in ["black", "blue", "brown", "green", "orange", "red", "violet", "white", "yellow"]:
            color_names.append(concept)
        elif orig_name in ["dotted", "striped", "zigzagged", "spiralled", "bubbled", "knitted", "chequered"]:
            texture_names.append(concept)
        else:
            concept_names.append(concept)
    concept_groups = []
    group_names = []
    group_types = []
    main_images = [
        "bicycle",
        "bicycle_Q11442",
        "bottle",
        "bottle_Q80228",
        "cat",
        "cat_Q146",
        "pottedplant",
        "pottedplant_Q27993793",
        "sheep",
        "sheep_Q7368",
    ]
    main_text = [
        "world",
        "world_1071646",
        "sports",
        "sports_650483",
        "business",
        "business_Q2142888",
        "science",
        "science_1505283",
    ]
    images = []
    text = []
    n_data_images = []
    n_data_text = []
    for concept in results_similarity.keys():
        split = concept.split("_")
        if split[-1] == "200":
            name = "_".join(split[:-1])
            if name in main_text:
                text.append(concept)
            if name in main_images:
                images.append(concept)
        if (
            "bottle_Q" in concept
            or "bicycle_Q" in concept
            or "cat_Q" in concept
            or "pottedplant_Q" in concept
            or "sheep_Q" in concept
        ):
            n_data_images.append(concept)
        if "world" in concept or "sports":
            n_data_text.append(concept)
    if cfg.model_name == "data2vec":
        images[-3:] = [images[-1], images[-3], images[-2]]
    n_data_images = [
        "bicycle_Q11442_30",
        "bottle_Q80228_30",
        "cat_Q146_30",
        "pottedplant_Q27993793_30",
        "sheep_Q7368_30",
        "bicycle_Q11442_50",
        "bottle_Q80228_50",
        "cat_Q146_50",
        "pottedplant_Q27993793_50",
        "sheep_Q7368_50",
        "bicycle_Q11442_200",
        "bottle_Q80228_200",
        "cat_Q146_200",
        "pottedplant_Q27993793_200",
        "sheep_Q7368_200",
        "bicycle_Q11442_1000",
        "bottle_Q80228_1000",
        "cat_Q146_1000",
        "pottedplant_Q27993793_1000",
        "sheep_Q7368_1000",
    ]
    for (
        group,
        name,
        g_type,
    ) in zip(
        [concept_names, color_names, texture_names, images, text, n_data_images, n_data_text],
        ["", "_colors", "_textures", "_images", "_text", "_sizes_images", "_sizes_text"],
        ["objectnet_vs_wiki", "objectnet_vs_wiki", "objectnet_vs_wiki", "objectnet_vs_wiki", "size", "size"],
    ):
        if len(group) > 0:
            concept_groups.append(group)
            group_names.append(name)
            group_types.append(g_type)
    for group, name, g_type in zip(concept_groups, group_names, group_types):
        for key, description, file in zip(
            ["cav car agreement", "cav accuracy", "car accuracy"],
            ["Agreement between CAV and CAR in %", "Accuracy of CAVs in %", "Accuracy of CARs in %"],
            [f"agreement{name}", f"acc_CAV{name}", f"acc_CAR{name}"],
        ):
            data_all = []
            names = []
            layers_all = []
            for concept in group:
                if g_type == "objectnet_vs_wiki":
                    if "_Q" in concept:
                        dataset = "Wikimedia"
                    else:
                        dataset = "Pascal VOC"
                    first = concept.split("_")[0]
                    names.append(f"{first} ({dataset})")
                else:
                    name_parts = concept.split("_")
                    names.append(f"{name_parts[0]} ({name_parts[-1]})")
                layers = list(results_similarity[concept].keys())
                layers = [l for l in layers if type(l) == int]
                layers_all.append(layers)
                results = [results_similarity[concept][layer][key] for layer in layers]
                if np.all([res <= 1 for res in results]):
                    results = [res * 100 for res in results]
                data_all.append(results)
            if "sizes" in name:
                plot_sizes(
                    data_all,
                    names,
                    layers_all,
                    y_label=description,
                    title=f"{description} - {cfg.model_name}",
                    path_save=f"{path_figures}{cfg.model_name}_{file}.png",
                )
            else:
                plot_lines(
                    data_all,
                    names,
                    layers_all,
                    y_label=description,
                    title=f"{description} - {cfg.model_name}",
                    path_save=f"{path_figures}{cfg.model_name}_{file}.png",
                )

        data_all_cav = []
        data_all_car = []
        names = []
        layers_all = []
        for concept in group:
            if g_type == "objectnet_vs_wiki":
                if "_Q" in concept:
                    dataset = "Wikimedia"
                else:
                    dataset = "Pascal VOC"
                first = concept.split("_")[0]
                names.append(f"{first} ({dataset})")
            else:
                name_parts = concept.split("_")
                names.append(f"{name_parts[0]} ({name_parts[-1]})")
            layers = list(results_similarity[concept].keys())
            layers = [l for l in layers if type(l) == int]
            layers_all.append(layers)
            results = [results_similarity[concept][layer]["cav accuracy"] for layer in layers]
            if np.all([res <= 1 for res in results]):
                results = [res * 100 for res in results]
            data_all_cav.append(results)

            results = [results_similarity[concept][layer]["car accuracy"] for layer in layers]
            if np.all([res <= 1 for res in results]):
                results = [res * 100 for res in results]
            data_all_car.append(results)
        if "sizes" in name:
            plot_sizes_pairs(
                [data_all_cav, data_all_car],
                names,
                layers_all,
                y_label="Accuracy in %",
                titles=["CAVs", "CARs"],
                path_save=f"{path_figures}{cfg.model_name}_acc_both{name}.png",
            )
        else:
            plot_labeled(
                [data_all_cav, data_all_car],
                names,
                layers_all,
                y_label="Accuracy in %",
                titles=["CAVs", "CARs"],
                path_save=f"{path_figures}{cfg.model_name}_acc_both{name}.png",
            )


if __name__ == "__main__":
    plot_similarity()
