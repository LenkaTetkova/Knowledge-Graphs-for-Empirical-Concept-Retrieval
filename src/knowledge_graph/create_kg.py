import matplotlib.pyplot as plt
import networkx as nx
from nltk.corpus import wordnet as wn
import requests
import numpy as np


def get_name(synset, source="wordnet"):
    """Get the name of a synset (without any prefixes/suffixes).

    :param synset: WordNet synset or ConceptNet concept
    :param source: Source of the synset, either "wordnet" or "conceptnet"
    :return: Name of the synset
    """
    if source == "wordnet":
        return synset.name().split(".")[0]
    elif source == "conceptnet":
        # todo
        return synset
    else:
        raise ValueError("Source not recognized, try wordnet or conceptnet")


def get_knowledge_graph_wn(synset, depth_up, depth_down):
    """Create a knowledge graph from a WordNet synset.

    :param synset: WordNet synset
    :param depth_up: Number of hypernyms to include
    :param depth_down: Number of hyponyms to include
    :return: Dictionary with the knowledge graph
    """
    # check that input is a synset
    # if not isinstance(synset, wn.synset):
    #    raise ValueError("Input must be a WordNet synset")

    # Create a dictionary to store the knowledge graph
    kg = {i: [] for i in range(depth_up, -depth_down - 1, -1)}
    kg[0] = [get_name(synset)]

    # Creat Graph
    G = nx.DiGraph()
    G.add_node(get_name(synset))

    # Recursively add hypernyms and their relationships up to the specified depth
    def add_hypernyms(synset, parent, current_depth, add_hyponyms=False):
        if current_depth <= depth_up:
            for hypernym in synset.hypernyms():
                hypernym_name = get_name(hypernym)
                G.add_node(hypernym_name)
                G.add_edge(hypernym_name, parent)
                kg[current_depth] = [hypernym_name]
                add_hypernyms(hypernym, hypernym_name, current_depth + 1)
                if add_hyponyms:
                    for hyponym in hypernym.hyponyms():
                        hyponym_name = get_name(hyponym)
                        kg[current_depth - 1].append(hyponym_name)
                        G.add_node(hyponym_name)
                        G.add_edge(parent, hyponym_name)

    # Recursively add hyponyms and their relationships up to the specified depth
    def add_hyponyms(synset, parent, current_depth):
        if current_depth <= depth_down:
            for hyponym in synset.hyponyms():
                hyponym_name = get_name(hyponym)
                G.add_node(hyponym_name)
                G.add_edge(parent, hyponym_name)
                kg[-current_depth].append(hyponym_name)
                add_hyponyms(hyponym, hyponym_name, current_depth + 1)

    # Add the hypernyms for each synset
    add_hypernyms(synset, get_name(synset), 1)
    add_hyponyms(synset, get_name(synset), 1)

    return kg, G


def get_edges(concept, relation):
    """Get edges from ConceptNet for a given concept and relation.

    :param concept: ConceptNet concept
    :param relation: Relation to the concept
    :return: List of edges  from ConceptNet
    """
    response = requests.get(
        f"http://api.conceptnet.io/query?node=/c/en/{concept}&rel=/r/{relation}&start=/c/en/{concept}"
    )
    edges = response.json().get("edges", [])
    all_edges = []
    for edge in edges:
        all_edges.append(
            edge["end"]["label"]
        )  # potential to add realtions towards the concept (use start instead of end in the query: edge['start']['label'])
    # clean names of concepts
    all_edges = clean_concepts(all_edges)
    # remove duplicates
    all_edges = list(set(all_edges))
    return all_edges


def clean_concepts(concepts):
    """Clean the concept to be used in ConceptNet.

    :param concept: Concept to be cleaned
    :return: Cleaned concept
    """
    cleaned_concepts = []
    for concept in concepts:
        # remove a and an from the concept
        if concept.startswith("a ") or concept.startswith("an "):
            concept = concept.replace("a ", "").replace("an ", "")
        # remove concepts with more than 3 words
        if len(concept.split()) > 3:
            continue
        cleaned_concepts.append(concept)
    return cleaned_concepts


def get_knowledge_graph_conceptnet(concept):
    """Create a knowledge graph from a ConceptNet concept.

    :param concept: ConceptNet concept
    :return: Dictionary with the knowledge graph
    """

    # define the realtions of interest, many others are available

    relations = ["IsA", "MadeOf", "HasA", "HasProperty", "PartOf"]
    kg = {i: [] for i in relations}

    # Add edges for the concept
    for relation in relations:
        edges = get_edges(concept, relation)
        kg[relation] = edges
    return kg


def visualize_kg(G, keyword):
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    plt.figure(figsize=(36, 12))
    nx.draw_networkx(
        G,
        pos,
        with_labels=True,
        node_size=200,
        node_color="lightblue",
        edge_color="gray",
        arrows=False,
    )
    plt.axis("off")
    plt.savefig(f"kg_{keyword}.png")
    plt.show()


def get_wikidata_id(dictionary):
    """Get the Wikidata ID for a given concept.

    :param dictionary: Dictionary with the knowledge graph
    :return: Wikidata ID
    """

    results = {i: [] for item in dictionary.values() for i in item}

    for item in results.keys():
        params = {"action": "wbsearchentities", "format": "json", "language": "en", "uselang": "en", "search": item}
        print(item)
        response = requests.get("https://www.wikidata.org/w/api.php?", params).json()
        search_results = response.get("search")
        if search_results:
            temp = []
            i = 0
            for result in search_results:
                result_id = result.get("id")
                result_description = result.get("description")
                print(i, result_id, result_description)
                i += 1
                temp.append({"id": result_id, "description": result_description})
            user_input = input("Which is the most relevant result? (input number or 'n' for none)")
            if user_input == "n":
                continue
            results[item] = temp[int(user_input)]["id"]
    print(results)
    return results


def create_kg(input, source, depth_up=1, depth_down=1):
    if source == "wordnet":
        kg = get_knowledge_graph_wn(input, depth_up, depth_down)
    elif source == "conceptnet":
        kg = get_knowledge_graph_conceptnet(input)
    else:
        raise ValueError("Source not recognized, try wordnet or conceptnet")
    return kg


# Call the function with the desired synset and depth
if __name__ == "__main__":
    synset = wn.synset("motor_vehicle.n.01")
    concept = "motor_vehicle"
    kg, G = create_kg(synset, "wordnet", 0, 2)
    print("Wordnet: ", kg)
    ids = get_wikidata_id(kg)
    np.save(f"src/data/wikidata_ids/{concept}.npy", ids)
    # visualize_kg(G, concept)
