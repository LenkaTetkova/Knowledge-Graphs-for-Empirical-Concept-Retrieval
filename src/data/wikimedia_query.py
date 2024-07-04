# Adapted from https://github.com/Redrrx/WikimediaScrap/blob/master/wikimedia_scraper.py
import hashlib
import json
import os
from datetime import datetime
from http.cookiejar import Cookie
from urllib.parse import urlparse

import hydra
import numpy as np
import requests
from omegaconf import DictConfig, OmegaConf
from tqdm.notebook import tqdm

from src.explanations.utils import path_edit

SPARQL_TEMPLATE_FOR_IMAGES = """
SELECT DISTINCT ?file ?image ?content_url WITH {{
      SELECT ?item ?itemLabel WHERE {{
        SERVICE <https://query.wikidata.org/sparql> {{
          {{?item wdt:P31/wdt:P279* wd:{concept}.}} UNION {{ BIND(wd:{concept} AS ?item) }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". ?item rdfs:label ?itemLabel . }}
        }}
        }}
}} AS %wikidataItems WHERE {{
  INCLUDE %wikidataItems .
  ?file wdt:P180 ?item;
        schema:url ?image;
        schema:contentUrl ?content_url .
 }}
LIMIT 100000
"""
#         wdt:P275 wd:Q6938433 .

SPARQL_TEMPLATE_FOR_TEXT = """

SELECT ?article ?name WHERE {{
  ?concept wdt:P31/wdt:P279* wd:{concept} .
  ?article schema:about ?concept ;
           schema:name ?name ;
           schema:isPartOf <https://en.wikipedia.org/> .
}}
"""

SPARQL_TEMPLATE_FOR_TEXT_MAIN = """

SELECT ?article ?name WHERE {{
  ?article schema:about wd:{concept} ;
           schema:name ?name ;
           schema:isPartOf <https://en.wikipedia.org/> .
}}
"""


def init_session(endpoint, token):
    """Initiate Wikimedia Commons Query Service.

    References
    ----------
    https://commons.wikimedia.org/wiki/Commons:SPARQL_query_service/API_endpoint

    """
    domain = urlparse(endpoint).netloc
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "name, e-mail",
        }
    )
    session.cookies.set_cookie(
        Cookie(0, "wcqsOauth", token, None, False, domain, False, False, "/", True, False, None, True, None, None, {})
    )
    return session


def generate_hashed_jpeg_filename(url):
    """
    Generates a filesystem-safe filename by hashing the URL or a part of it.
    """
    # Extract the filename or a unique part of the URL
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)

    # Hash the filename using SHA-256
    hash_digest = hashlib.sha256(filename.encode()).hexdigest() + ".jpeg"

    # Ensure the hash string length is within filesystem limits
    # SHA-256 produces a 64 characters long hash, well within the 255 character limit
    return hash_digest


def download_file(url, target_dir):
    """
    Downloads a file from a URL, generates a safe filename, and saves it to a target directory.
    """
    # Generate a safe filename
    filename = generate_hashed_jpeg_filename(url)

    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Define the full path for the file to be saved
    file_path = os.path.join(target_dir, filename)

    if os.path.isfile(file_path):
        # print(f"File already saved as {file_path}")
        return

    # Download and save the file
    response = requests.get(url, headers={"User-Agent": "name, e-mail"})
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        # print(f"File saved as: {file_path}")
    else:
        print("Failed to download the file: {status_code}".format(status_code=response.status_code))


def download_640px_file(url, target_dir):
    """
    Downloads a file from a URL, generates a safe filename, and saves it to a target directory.
    """
    # Change URL to thumb URL
    # url = "https://commons.wikimedia.org/wiki/Special:Random/Image"
    url_parts = url.split("/")
    url = "/".join(url_parts[:5]) + "/thumb/" + "/".join(url_parts[5:]) + "/" + "640px-" + url_parts[-1]

    # Generate a safe filename
    filename = generate_hashed_jpeg_filename(url)

    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Define the full path for the file to be saved
    file_path = os.path.join(target_dir, filename)

    if os.path.isfile(file_path):
        # print(f"File already saved as {file_path}")
        return

    # Download and save the file
    response = requests.get(url, headers={"User-Agent": "name, e-mail"})

    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        # print(f"File saved as: {file_path}")
    else:
        print("Failed {status_code} to download the file: {url}".format(status_code=response.status_code, url=url))


@hydra.main(config_path="../config", config_name="default.yaml")
def query(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    orig_cwd = hydra.utils.get_original_cwd()

    path_data = path_edit(cfg.path_data, orig_cwd)

    if not os.path.exists(f"{path_data}{cfg.data_type}"):
        os.makedirs(f"{path_data}{cfg.data_type}")

    concepts = [
        "Q503",  # banana
        "Q201097",  # basket
        "Q11032",  # newspaper
        "Q446",  # wheel
        "Q200539",  # dress
        # For AG News
        "Q1071646",  # world news, the item world is "Q16502",
        "Q650483",  # sports journalism, the item sport is  "Q349",
        "Q2142888",  # business journalism, the item business is "Q4830453",
        "Q1505283",  # science journalism, the item science is "Q336",
    ]

    for concept in ["motor_vehicle", "sport", "fruit", "edible_fruit"]:
        subconcepts = np.load(f"{orig_cwd}/src/data/wikidata_ids/{concept}.npy", allow_pickle=True).item()

        for name, wiki_id in subconcepts.items():
            if wiki_id is not None and type(wiki_id) is not list:
                concepts.append(wiki_id)
    if cfg.random:
        concepts = []
        for name in ["Entity", "Human", "Nature", "Earth"]:
            query = f"https://en.wikipedia.org/w/api.php?action=query&generator=links&titles={name}&prop=pageprops&ppprop=wikibase_item&gpllimit=500&format=json"
            response = requests.get(query, headers={}, params={})
            response_json = json.loads(response.text)
            for page in response_json["query"]["pages"].values():
                try:
                    concepts.append(page["pageprops"]["wikibase_item"])
                except KeyError:
                    continue
    if cfg.data_type == "images":
        ENDPOINT = "https://commons-query.wikimedia.org/sparql"

        session = init_session(ENDPOINT, "be36729232cb4775a7c73a493e76c496.4efd47c00877c8f4ba12c12d2dc78b1fc01fedcb")

        for concept in tqdm(concepts):
            print(concept)
            QUERY_RESULT_FILENAME = f"query-result_{concept}.csv"
            now = datetime.now().isoformat()
            sparql = SPARQL_TEMPLATE_FOR_IMAGES.format(concept=concept)
            response = session.post(url=ENDPOINT, data={"query": sparql}, headers={"Accept": "application/json"})
            response.raise_for_status()
            with open(QUERY_RESULT_FILENAME, "a") as fid:
                json_string = json.dumps(response.json())
                fid.write(f"{concept},{now},{json_string}\n")

            n_images = 0
            if cfg.random:
                target_dir = f"{path_data}{cfg.data_type}/random"
            else:
                target_dir = f"{path_data}{cfg.data_type}/{concept}"
            for line in tqdm(open(QUERY_RESULT_FILENAME, "r")):
                concept, query_datetime, json_string = line.split(",", 2)
                data = json.loads(json_string)
                for row in tqdm(data["results"]["bindings"]):
                    url = row["content_url"]["value"]
                    print(url)
                    download_640px_file(url, target_dir)
                    n_images += 1
            print(f"Concept {concept}: {n_images} images.")

    elif cfg.data_type == "text":
        ENDPOINT = "https://query.wikidata.org/sparql"
        EXTRACTS_API_URL_TEMPLATE = "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&titles={titles}&formatversion=2&explaintext=1&exsectionformat=plain"
        USER_AGENT = "Name, e-mail"

        for concept in tqdm(concepts):
            if cfg.random:
                EXTRACT_RESULT_FILENAME = f"{path_data}{cfg.data_type}/extract-result_random.csv"
            else:
                EXTRACT_RESULT_FILENAME = f"{path_data}{cfg.data_type}/extract-result_{concept}.csv"
            sparql_main = SPARQL_TEMPLATE_FOR_TEXT_MAIN.format(concept=concept)
            response = requests.post(
                ENDPOINT, data={"query": sparql_main}, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
            )
            response.raise_for_status()
            if response.status_code == 200:
                response_data = response.json()
                try:
                    bind = response_data["results"]["bindings"][0]
                except:
                    print(f"Concept {concept} not found.")
                    continue
                title = bind["name"]["value"]
                url = EXTRACTS_API_URL_TEMPLATE.format(titles=title)
                api_response = requests.get(url, headers={"User-Agent": USER_AGENT})
                api_response.raise_for_status()
                if api_response.status_code == 200:
                    now = datetime.now().isoformat()
                    api_response_data = api_response.json()
                    try:
                        if len(api_response_data["query"]["pages"][0]["extract"]) > 0:
                            json_string = json.dumps(api_response_data["query"]["pages"][0])
                            with open(EXTRACT_RESULT_FILENAME, "a") as fid:
                                fid.write(f"{concept},{now},{json_string}\n")
                    except KeyError as e:
                        print(f"Title {concept}")
                        raise e
            n_subconcepts = 0

            sparql = SPARQL_TEMPLATE_FOR_TEXT.format(concept=concept)
            response = requests.post(
                ENDPOINT, data={"query": sparql}, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
            )
            response.raise_for_status()
            if response.status_code == 200:
                response_data = response.json()
                bindings = response_data["results"]["bindings"]
                for bind in bindings:
                    title = bind["name"]["value"]
                    url = EXTRACTS_API_URL_TEMPLATE.format(titles=title)
                    api_response = requests.get(url, headers={"User-Agent": USER_AGENT})
                    api_response.raise_for_status()
                    if api_response.status_code == 200:
                        now = datetime.now().isoformat()
                        api_response_data = api_response.json()
                        try:
                            if len(api_response_data["query"]["pages"][0]["extract"]) > 0:
                                json_string = json.dumps(api_response_data["query"]["pages"][0])
                                with open(EXTRACT_RESULT_FILENAME, "a") as fid:
                                    fid.write(f"{title},{now},{json_string}\n")
                                n_subconcepts += 1
                        except KeyError as e:
                            print(f"Title {title}")
            print(f"Concept {concept}: {n_subconcepts} subconcepts.")
    else:
        raise NotImplementedError


if __name__ == "__main__":
    query()
