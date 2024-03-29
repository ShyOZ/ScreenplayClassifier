import re
import json
from pathlib import Path

import requests
from tqdm import tqdm
import multiprocessing
from bs4 import BeautifulSoup
from typing import Set, Optional

from script_info import ScriptInfo

import constants

IMSDB_ROOT = "http://www.imsdb.com"


def get_movie_info_links_from_genre(genre: str) -> Optional[Set[str]]:
    request = requests.get(f"http://www.imsdb.com/genre/{genre}", timeout=5)
    if request.status_code != requests.codes.ok:
        return None
    request_text = request.text
    soup = BeautifulSoup(request_text, "html5lib")

    link_tags = soup.find_all("a", href=True, title=re.compile("Script$"))

    return {IMSDB_ROOT + tag["href"] for tag in link_tags}


def standardize_title(title: str) -> str:
    the_idx = title.find(", The")
    if the_idx != -1:
        title = "The " + title[:the_idx] + title[the_idx + 5:]
    a_idx = title.find(", A")
    if a_idx != -1:
        title = "A " + title[:a_idx] + title[a_idx + 3:]
    return title.replace(":", " -")


def as_filename_compatible(title: str) -> str:
    return re.sub(r"[\\/:*?\"<>|.]+", "", standardize_title(title))


def get_script_information(movie_info_url: str) -> Optional[ScriptInfo]:
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

    if "Sci-Fi" in genres:
        genres.remove("Sci-Fi")
        genres.add("SciFi")

    genres = set(filter(lambda g: g in constants.GENRE_LABELS, genres))

    return ScriptInfo(standardize_title(title), as_filename_compatible(title), movie_info_url, script_url, genres)


def get_movie_script(script_url: str) -> Optional[str]:
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


def write_script_to_file(script_filename: str, script_text: str) -> None:
    script_file_path = Path(constants.TRAIN_SCREENPLAYS_PATH / script_filename).with_suffix(".txt")
    script_file_path.write_text(script_text, encoding="utf-8")


def process_script_info(script_info: ScriptInfo) -> Optional[dict]:
    while True:
        try:
            script_text = get_movie_script(script_info.script_url)

            if script_text is None:
                return None

            if len(script_text) < 3000:  # 3kb
                return None

            write_script_to_file(script_info.filename, script_text)

            return script_info.to_dict()

        except (requests.ConnectionError, requests.Timeout):
            pass


def process_info_url(info_url: str) -> Optional[dict]:
    while True:
        try:
            script_info = get_script_information(info_url)
            if script_info is None:
                return None

            return process_script_info(script_info)

        except (requests.ConnectionError, requests.Timeout):
            pass


def scrape_from_scratch():
    all_urls = set()

    for genre in tqdm(constants.GENRE_LABELS, desc="processing genres"):
        genre_urls = get_movie_info_links_from_genre(genre)
        all_urls = all_urls.union(genre_urls)

    with multiprocessing.Pool(4) as pool:
        movie_info = list(
            tqdm(pool.imap(process_info_url, all_urls), total=len(all_urls), desc="processing script info"))

    movie_info = filter(lambda x: x is not None, movie_info)

    movie_info = sorted(movie_info, key=lambda info: info["title"])

    with constants.MOVIE_INFO_PATH.open("w") as f:
        json.dump(movie_info, f)


def scrape_from_existing():
    movie_info = ScriptInfo.schema().loads(constants.MOVIE_INFO_PATH.read_text(), many=True)

    with multiprocessing.Pool(4) as pool:
        list(tqdm(pool.imap(process_script_info, movie_info), total=len(movie_info), desc="processing script info"))


if __name__ == "__main__":
    constants.TRAIN_SCREENPLAYS_PATH.mkdir(parents=True, exist_ok=True)

    if constants.MOVIE_INFO_PATH.exists():
        scrape_from_existing()
    else:
        scrape_from_scratch()
