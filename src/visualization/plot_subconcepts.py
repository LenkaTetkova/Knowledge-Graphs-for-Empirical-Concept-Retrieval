import pickle
import random

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.stats import sem

from src.explanations.utils import path_edit


def plot_subconcepts_all(results, group_names, layers, model_name, path_figures, type_name):
    if type_name == "triplets":
        description = f"Proportion of closest concepts in the triplets test"
    elif type_name == "average":
        description = f"CAV: Cosine similarity of concepts"
    elif type_name == "cav":
        description = f"CAV: Percentage of subconcept classified as concept"
    elif type_name == "car":
        description = f"CAR: Percentage of subconcept classified as concept"
    file = f"tuples_{type_name}"

    if type_name in ["triplets", "cav", "car"]:
        l_limit = 0
    else:
        l_limit = -1
    if type_name in ["cav", "car"]:
        u_limit = 105
    else:
        u_limit = 1
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
    fig, axs = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(9, 9))
    positions = {"bert": [0, 0], "roberta": [0, 1], "data2vec": [1, 0], "vit": [1, 1]}
    for name_pretrained, name_finetuned, nice_name in zip(
        ["bert", "roberta", "data2vec", "vit"],
        ["bert_finetuned", "roberta_go", "data2vec_finetuned", "vit_finetuned"],
        ["BERT", "RoBERTa", "data2vec", "ViT"],
    ):
        pos_x = positions[name_pretrained][0]
        pos_y = positions[name_pretrained][1]
        for i, group_name in enumerate(["sport", "fruit", "motor vehicle", "non-related"]):
            axs[pos_x, pos_y].plot(
                layers,
                results[name_pretrained][i],
                f"o-",
                label=f"{group_name} (pretrained)",
            )
            try:
                axs[pos_x, pos_y].plot(
                    layers,
                    results[name_finetuned][i],
                    f"o-",
                    label=f"{group_name} (fine-tuned)",
                )
            except:
                axs[pos_x, pos_y].plot(
                    [],
                    [],
                    f"o-",
                    label=f"{group_name} (fine-tuned)",
                )
                "error"
            axs[pos_x, pos_y].set_title(nice_name)
            axs[pos_x, pos_y].set_xticks(list(range(13)))
            if u_limit > 2:
                axs[pos_x, pos_y].set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            axs[pos_x, pos_y].grid(True, which="both")
            axs[pos_x, pos_y].set_xlabel("Layers")
            axs[pos_x, pos_y].set_ylim([l_limit, u_limit])
    plt.rcParams.update(rcParams)
    plt.xticks(list(range(13)))
    if u_limit > 2:
        plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.grid(True, which="both")
    lgd = plt.legend(ncol=4, fontsize=10, bbox_to_anchor=(-0.1, -0.2), loc="upper center")
    plt.savefig(f"{path_figures}/{file}.png", dpi=600, bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.close()
    return


@hydra.main(config_path="../config", config_name="default.yaml")
def plot_subconcepts(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    orig_cwd = hydra.utils.get_original_cwd()
    # Fix random seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    path_outputs = path_edit(cfg.path_outputs, orig_cwd)
    path_figures = path_edit("./reports/figures/", orig_cwd)

    groups = {}
    group_id = 0
    for concept in ["sport", "edible_fruit", "motor_vehicle"]:
        subconcepts = np.load(f"{orig_cwd}/src/data/wikidata_ids/{concept}.npy", allow_pickle=True).item()

        for name, wiki_id in subconcepts.items():
            if wiki_id is not None and type(wiki_id) is not list:
                groups[wiki_id] = group_id
        group_id += 1

    group_names = ["sport", "fruit", "motor vehicle", "various"]
    data_types = ["text", "text", "images", "images", "text", "text", "images", "images"]
    types_names = ["cav", "car"]

    results_final_cav = {}
    results_final_car = {}
    model_names = [
        "bert",
        "roberta",
        "data2vec",
        "vit",
        "bert_finetuned",
        "roberta_go",
        "data2vec_finetuned",
        "vit_finetuned",
    ]
    for model_name, data_type in zip(model_names, data_types):
        path_results = f"{path_outputs}{data_type}_{model_name}_pairs.pkl"

        with open(path_results, "rb") as fp:
            results_raw = pickle.load(fp)
        results_cav = {}
        results_car = {}

        groups_keys = [[] for k in range(4)]
        for key in results_raw.keys():
            ids = key.split("_")
            if groups[ids[0]] != groups[ids[1]]:
                groups_keys[3].append(key)
            elif groups[ids[0]] == 0 and groups[ids[1]] == 0:
                groups_keys[0].append(key)
            elif groups[ids[0]] == 1 and groups[ids[1]] == 1:
                groups_keys[1].append(key)
            else:
                groups_keys[2].append(key)

        for key in results_raw.keys():
            results_cav[key] = {}
            results_car[key] = {}
            for layer in results_raw[key].keys():
                if layer not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                    continue
                results_cav[key][layer] = results_raw[key][layer]["cav"]
                if results_raw[key][layer]["cav"] == results_raw[key][layer]["car"]:
                    try:
                        results_car[key][layer] = results_raw[key][key]["car"][layer]
                    except KeyError as e:
                        results_car[key][layer] = results_raw[key][layer]["car"]
                else:
                    results_car[key][layer] = results_raw[key][layer]["car"]

        for key, results in zip(["cav", "car"], [results_cav, results_car]):
            means = {}
            sems = {}
            layers_union = []
            for group, name in zip(groups_keys, group_names):
                if len(group) == 0:
                    continue
                group_means = {}
                group_sems = {}
                for layer in results[group[0]].keys():
                    all_results = []
                    for concept in group:
                        try:
                            all_results.append(results[concept][layer])
                        except KeyError:
                            print(f"Concept {concept} not found")
                    group_means[layer] = np.mean(all_results)
                    group_sems[layer] = sem(all_results)
                    if layer not in layers_union:
                        layers_union.append(layer)
                means[name] = group_means
                sems[name] = group_sems

                data_all = []
                names = []
                layers_all = []
                for concept in group:
                    try:
                        layers = list(results[concept].keys())
                        data_all.append([results[concept][layer] for layer in layers])
                        names.append(concept)
                        layers_all.append(layers)
                    except KeyError:
                        print(f"Concept {concept} not found")
            sports = [means["sport"][layer] for layer in layers_union]
            fruits = [means["fruit"][layer] for layer in layers_union]
            try:
                motor_vehicles = [means["motor vehicle"][layer] for layer in layers_union]
            except:
                print("Motor vehicles not loaded")
                motor_vehicles = []
            various = [means["various"][layer] for layer in layers_union]
            if key == "cav":
                results_final_cav[model_name] = [sports, fruits, motor_vehicles, various]
            else:
                results_final_car[model_name] = [sports, fruits, motor_vehicles, various]

    for results, type_name in zip([results_final_cav, results_final_car], types_names):
        plot_subconcepts_all(results, group_names, layers_union, cfg.model_name, path_figures, type_name)


if __name__ == "__main__":
    plot_subconcepts()
