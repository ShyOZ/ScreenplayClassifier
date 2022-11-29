import re
import json
import requests
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from typing import Set, Union

IMSDB_ROOT = "http://www.imsdb.com"

RESOURCE_PATH = Path("Resources")
OUTPUT_PATH = RESOURCE_PATH / "TrainScreenplays"
TARGET_GENRES = set((RESOURCE_PATH / "Genres.txt").read_text().splitlines())

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

@dataclass_json
@dataclass(frozen=True)
class ScriptInfo:
    title: str
    info_url: str
    script_url: str
    genres: set[str] = field(default_factory=set)

def get_movie_info_links_from_genre(genre: str) -> Union[Set[str], None]:
    request = requests.get(f"http://www.imsdb.com/genre/{genre}", timeout=5)
    if request.status_code != requests.codes.ok:
        return None
    request_text = request.text
    soup = BeautifulSoup(request_text, "html5lib")

    link_tags = soup.find_all("a", href=True, title=re.compile("Script$"))

    return {IMSDB_ROOT + tag["href"] for tag in link_tags}


def get_script_information(movie_info_url: str) -> Union[ScriptInfo, None]:
    request = requests.get(movie_info_url, timeout=5)
    if request.status_code != requests.codes.ok:
        return None

    request_text = request.text
    soup = BeautifulSoup(request_text, "html5lib")
    details_table = soup.find("table", class_="script-details")
    if not details_table:
        return None

    script_link = details_table.find(
        "a", href=re.compile("^/scripts/.+\.html"))

    if not script_link:
        return None

    title = re.search('\".+\"', script_link.text)[0][1:-1]

    script_url = IMSDB_ROOT + script_link["href"]

    genre_links = details_table.find_all("a", title=re.compile("Scripts$"))
    genres = {genre["href"][len("/genre/"):] for genre in genre_links}
    genres = set(filter(lambda g: g in TARGET_GENRES, genres))

    return ScriptInfo(title, movie_info_url, script_url, genres)


def get_movie_script(script_url: str) -> Union[str, None]:
    request = requests.get(script_url, timeout=5)
    if request.status_code != requests.codes.ok:
        return None
    request_text = request.text
    soup = BeautifulSoup(request_text, "html5lib")

    pre_tags = soup.find_all("pre")
    if pre_tags:
        return pre_tags[-1].text

    scrtext_tags = soup.find_all("td", class_="scrtext")
    if scrtext_tags:
        return scrtext_tags[-1].text

    return None


def write_script_to_file(script_title: str, script_text: str) -> None:
    script_title = re.sub(r"[^\w\-_. ]+", "_", script_title)
    script_file_path = OUTPUT_PATH / (script_title + ".txt")
    script_file_path.write_text(script_text, encoding="utf-8")


def process_task(info_url):
    while True:
        try:
            script_info = get_script_information(info_url)
            if script_info is None:
                return None

            script_text = get_movie_script(script_info.script_url)

            if script_text is None:
                return None

            write_script_to_file(script_info.title, script_text)

            return script_info.to_dict()

        except (requests.ConnectionError, requests.Timeout):
            pass


if __name__ == "__main__":
    all_urls = set()

    for genre in tqdm(TARGET_GENRES, desc="processing genres"):
        genre_urls = get_movie_info_links_from_genre(genre)
        all_urls = all_urls.union(genre_urls)

    with multiprocessing.Pool(4) as pool:
        movie_info = list(tqdm(pool.imap(process_task, all_urls), total=len(all_urls), desc="processing script info"))

    movie_info = filter(lambda x: x is not None, movie_info)

    movie_info = sorted(movie_info, key=lambda info: info["title"])

    with open(RESOURCE_PATH / "Movie Script Info.json", "w") as f:
        json.dump(movie_info, f)
