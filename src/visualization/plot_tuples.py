import pickle
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.stats import sem

from src.explanations.utils import path_edit
from src.visualization.plot_similarity import plot_lines


def plot_subconcepts_tuples(results, random_cosine, concept_groups, group_names, model_name, path_figures, type_name):
    if len(list(random_cosine.keys())) > 0:
        random_upper = []
        random_lower = []
        cosine_keys = list(random_cosine.keys())
        one_concept = cosine_keys[0]
        for layer in random_cosine[one_concept].keys():
            means = [random_cosine[conc][layer][0] for conc in cosine_keys]
            vars = [random_cosine[conc][layer][1] for conc in cosine_keys]
            mean_var = np.mean(vars)
            mean_mean = np.mean(means)
            sqrt_var = np.sqrt(mean_var)
            random_upper.append(mean_mean + sqrt_var)
            random_lower.append(mean_mean - sqrt_var)
    if type_name == "triplets":
        description = f"Proportion of closest concepts in the triplets test"
    elif type_name == "average":
        description = f"CAV: Cosine similarity of concepts"
    elif type_name == "cav":
        description = f"CAV: Percentage of subconcept classified as concept"
    elif type_name == "car":
        description = f"CAR: Percentage of subconcept classified as concept"
    file = f"tuples_{type_name}"

    means = {}
    sems = {}
    layers_union = []
    for group, name in zip(concept_groups, group_names):
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
        if len(list(random_cosine.keys())) > 0:
            names.append("random")
            layers_all.append(list(random_cosine[one_concept].keys()))
            data_all.append(
                {
                    "upper": random_upper,
                    "lower": random_lower,
                }
            )

    print(f"{model_name}: {type_name}")
    if means["sport"][layer] > 1:
        for layer in layers_union:
            print(
                f"Layer {layer} \t & {means['sport'][layer]:.1f} "
                f"$\\pm$ {sems['sport'][layer]:.1f} \t "
                f"& {means['fruit'][layer]: .1f}"
                f"$\\pm$ {sems['fruit'][layer]:.1f} \t "
                f"& {means['motor vehicle'][layer]:.1f}"
                f"$\\pm$ {sems['motor vehicle'][layer]:.1f} \t "
                f"& {means['various'][layer]:.1f}"
                f"$\\pm$ {sems['various'][layer]:.1f} \t \\\\ \\hline"
            )
    else:
        for layer in layers_union:
            print(
                f"Layer {layer} \t & {means['sport'][layer]:.3f} "
                f"$\\pm$ {sems['sport'][layer]:.3f} \t "
                f"& {means['fruit'][layer]: .3f}"
                f"$\\pm$ {sems['fruit'][layer]:.3f} \t "
                f"& {means['motor vehicle'][layer]: .3f}"
                f"$\\pm$ {sems['motor vehicle'][layer]:.3f} \t "
                f"& {means['various'][layer]: .3f}"
                f"$\\pm$ {sems['various'][layer]:.3f} \t \\\\ \\hline"
            )
    sports = [means["sport"][layer] for layer in layers_union]
    fruits = [means["fruit"][layer] for layer in layers_union]
    motor_vehicles = [means["motor vehicle"][layer] for layer in layers_union]
    various = [means["various"][layer] for layer in layers_union]
    if type_name in ["triplets", "cav", "car"]:
        l_limit = 0
    else:
        l_limit = -1
    if type_name in ["cav", "car"]:
        u_limit = 105
    else:
        u_limit = 1
    plot_lines(
        [sports, fruits, motor_vehicles, various],
        ["sport", "fruit", "motor vehicle", "various"],
        [layers_union for i in range(4)],
        y_label=description,
        title=f"{description} - {model_name}",
        path_save=f"{path_figures}/{model_name}_{file}.png",
        upper_limit=u_limit,
        lower_limit=l_limit,
    )
    if np.mean(sports) > 1:
        print(
            f"Average \t & {np.mean(sports):.1f} "
            f"$\\pm$ {sem(sports):.1f} \t "
            f"& {np.mean(fruits): .1f}"
            f"$\\pm$ {sem(fruits):.1f} \t "
            f"& {np.mean(motor_vehicles): .1f}"
            f"$\\pm$ {sem(motor_vehicles):.1f} \t "
            f"& {np.mean(various): .1f}"
            f"$\\pm$ {sem(various):.1f} \t \\\\ \\hline"
        )
    else:
        print(
            f"Average \t & {np.mean(sports):.3f} "
            f"$\\pm$ {sem(sports):.3f} \t "
            f"& {np.mean(fruits): .3f}"
            f"$\\pm$ {sem(fruits):.3f} \t "
            f"& {np.mean(motor_vehicles): .3f}"
            f"$\\pm$ {sem(motor_vehicles):.3f} \t "
            f"& {np.mean(various): .3f}"
            f"$\\pm$ {sem(various):.3f} \t \\\\ \\hline"
        )
    return


@hydra.main(config_path="../config", config_name="default.yaml")
def plot_tuples(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    orig_cwd = hydra.utils.get_original_cwd()
    # Fix random seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    path_outputs = path_edit(cfg.path_outputs, orig_cwd)
    path_figures = path_edit("./reports/figures/", orig_cwd)

    path_tuples = f"{path_outputs}{cfg.data_type}_{cfg.model_name}_tuples.pkl"
    with open(path_tuples, "rb") as fp:
        results_similarity = pickle.load(fp)
        try:
            random_cosine = pickle.load(fp)
        except:
            random_cosine = {}
        try:
            results_triplets = pickle.load(fp)
        except:
            results_triplets = {}

    groups = {}
    group_id = 0
    for concept in ["sport", "edible_fruit", "motor_vehicle"]:
        subconcepts = np.load(f"{orig_cwd}/src/data/wikidata_ids/{concept}.npy", allow_pickle=True).item()

        for name, wiki_id in subconcepts.items():
            if wiki_id is not None and type(wiki_id) is not list:
                groups[wiki_id] = group_id
        group_id += 1

    groups_keys = [[] for k in range(4)]
    for key in results_similarity.keys():
        ids = key.split("_")
        if groups[ids[0]] != groups[ids[1]]:
            groups_keys[3].append(key)
        elif groups[ids[0]] == 0 and groups[ids[1]] == 0:
            groups_keys[0].append(key)
        elif groups[ids[0]] == 1 and groups[ids[1]] == 1:
            groups_keys[1].append(key)
        else:
            groups_keys[2].append(key)

    group_names = ["sport", "fruit", "motor vehicle", "various"]
    results_groups = [results_similarity]
    types_names = ["average"]
    random_results = [random_cosine]
    if len(list(results_triplets.keys())) > 0:
        results_groups.append(results_triplets)
        types_names.append("triplets")
        random_results.append({})
    for results, type_name, cosine in zip(results_groups, types_names, random_results):
        plot_subconcepts_tuples(results, cosine, groups_keys, group_names, cfg.model_name, path_figures, type_name)


if __name__ == "__main__":
    plot_tuples()
