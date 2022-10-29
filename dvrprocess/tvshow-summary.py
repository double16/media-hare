#!/usr/bin/env python3

import datetime
import getopt
import hashlib
import json
import os
import random
import sys
import time
import xml.etree.ElementTree as ET

import imdb
import requests
from imdb import IMDbDataAccessError

import common

#
# List of libraries: /library/sections, want type="show"
# Shows for a section: /library/sections/2/all from Directory.key in section
# All Episodes: /library/metadata/83179/allLeaves from Directory.key in show
#

# Produce a structure from the API:
#   dict: key=show title, value=list of tuple (library, show title, year, season, episode, duration in minutes, video_codec, audio_codec, videoResolution, bitrate)
#

ia = imdb.IMDb()


def usage():
    print(f"""
Produce two reports on TV episodes. One is a CSV of episode data such as length, codecs, bit rate, etc. The second is
a report of missing episodes and seasons.

Usage: {sys.argv[0]} -u http://127.0.0.1:32400 -e episodes.csv -c completion.csv
"url=", "episodes=", "completion=", "cache_dir="

-u, --url=
    The Plex Media Server URL. Specify as http://127.0.0.1:32400, default is {common.get_plex_url()}
-e, --episodes=
    Location of the episode CSV output file.
-c, --completion=
    Location of the completion CSV output file.
-t, --cache_dir=
    Cache directory for IMDb data.
""", file=sys.stderr)


def read_from_api(plex_url, limit=None):
    result = {}
    sections_response = requests.get(f'{plex_url}/library/sections')
    sections = list(
        filter(lambda el: el.tag == 'Directory' and el.attrib['type'] == 'show' and 'DVR' not in el.attrib['title'],
               ET.fromstring(sections_response.text)))

    for section in sections:
        section_response = requests.get(
            f'{plex_url}/library/sections/{section.attrib["key"]}/all')
        shows = list(filter(lambda el: el.tag ==
                                       'Directory' and el.attrib['type'] == 'show',
                            ET.fromstring(section_response.text)))
        for show in shows:
            show_response = requests.get(
                f'{plex_url}{show.attrib["key"].replace("/children", "/allLeaves")}')
            episodes = list(filter(
                lambda el: el.tag == 'Video' and el.attrib['type'] == 'episode' and 'index' in el.attrib,
                ET.fromstring(show_response.text)))
            episode_list = []
            for episode in episodes:
                episode_num = int(episode.attrib["index"])
                season = int(episode.attrib["parentIndex"])
                duration = episode.attrib.get("duration")
                if duration is not None:
                    duration = int(int(duration) / 60000)
                # title = episode.attrib["title"]

                video_codec = "?"
                audio_codec = "?"
                video_resolution = "?"
                bitrate = "?"
                framerate = "?"
                file_size = "?"
                for media in list(filter(lambda el: el.tag == 'Media', episode)):
                    video_codec = media.attrib.get("videoCodec", "?")
                    audio_codec = media.attrib.get("audioCodec", "?")
                    video_resolution = media.attrib.get("videoResolution", "?")
                    framerate = common.frame_rate_from_s(media.attrib.get("videoFrameRate"))
                    bitrate = media.attrib.get("bitrate", "?")
                    for part in list(filter(lambda el: el.tag == 'Part', media)):
                        file_size = int(part.attrib.get("size", "0"))

                t = (section.attrib["title"], show.attrib["title"], show.attrib.get("year"),
                     season, episode_num, duration, video_codec, audio_codec,
                     video_resolution, bitrate, framerate, file_size)
                episode_list.append(t)

            if len(episode_list) > 0:
                result[show.attrib["title"]] = episode_list
                if limit and len(result) >= limit:
                    return result

    return result


def episode_human_sort_key(e):
    return e[0][1].replace("The ", "")


def episode_human_sort(seq):
    l = list(seq)
    l.sort(key=episode_human_sort_key)
    return l


# Print a CSV of all shows, all episodes
def print_episode_csv(shows, fileno=sys.stdout):
    fileno.write(
        '"Show","Section","Year","Season","Episode","Length","Video Codec","Audio Codec","Resolution","Frame Rate","Bit Rate","Size"\n')
    for show in episode_human_sort(shows.values()):
        for episode in show:
            fileno.write(
                f'"{episode[1]}","{episode[0]}","{episode[2]}","{episode[3]}","{episode[4]}",{episode[5]}'
                f',"{episode[6]}","{episode[7]}","{episode[8]}","{episode[10]}",{episode[9]},{episode[11]}\n')


def format_season(season):
    return "S" + str(season).zfill(2)


def format_episode(season, episode):
    return "S" + str(season).zfill(2) + "E" + str(episode).zfill(2)


def hash_str_sha256(array):
    str_array = list(filter(lambda e: e is not None, array))
    return hashlib.sha256(','.join(str_array).encode('utf-8')).hexdigest()


def read_cached_json(cache_dir, cache_filename, consider_expiry=True):
    cache_file = f"{cache_dir}/{cache_filename}"
    if not os.path.isfile(cache_file):
        return None
    cached_data = json.loads(open(cache_file).read())
    if not consider_expiry:
        return cached_data

    seconds_in_day = 86400
    expiry_days = 30
    expiry_span_days = 2
    if ('year' in cached_data) and int(cached_data['year']) > 1900 and 'expected' in cached_data:
        last_year = int(cached_data['year']) + int(max(cached_data['expected']['seasons']))
        if last_year < (datetime.datetime.now().date().year - 2):
            expiry_days = 180
            expiry_span_days = 60

    # print(f"INFO: expiry days {expiry_days}, expiry span {expiry_span_days}")
    if (time.time() - os.path.getmtime(cache_file)) > (
            expiry_days * seconds_in_day + random.randint(-expiry_span_days * seconds_in_day,
                                                          expiry_span_days * seconds_in_day)):
        return None

    return cached_data


def write_cached_json(cache_dir: str, cache_filename: str, data):
    cache_file = os.path.join(cache_dir, cache_filename)
    with open(cache_file, "w") as outfile:
        outfile.write(json.dumps(data, indent=4))


def imdb_get_movie_id(show, cache_dir: str):
    if cache_dir is None:
        return None

    name = show[0][1]
    year = show[0][2]
    # skip if year is not present, makes matching unreliable
    if not year:
        return None
    filename = f"show-{hash_str_sha256([name, year])}.json"
    cached_data = read_cached_json(cache_dir, filename)
    if cached_data:
        return cached_data.get('imdb_id')

    if year:
        search_text = f"{name} ({year})"
    else:
        search_text = name
    # print(f"INFO: Calling IMDb for {search_text}")
    results = ia.search_movie(search_text, 1)
    if not results or len(results) == 0:
        movie_id = None
    else:
        movie_id = results[0].movieID
    cached_data = {
        'imdb_id': movie_id,
        'name': name,
        'year': year,
    }
    write_cached_json(cache_dir, filename, cached_data)

    return movie_id


def infer_show_stats(show):
    season_max = 0
    episode_max = 0
    for e in show:
        season_max = max(season_max, e[3])
        episode_max = max(episode_max, e[4])
    result = {}
    episode_list = []
    for x in range(1, episode_max + 1):
        episode_list.append(x)
    seasons = []
    episodes = {}
    for x in range(1, season_max + 1):
        seasons.append(x)
        episodes[str(x)] = episode_list

    result['seasons'] = seasons
    result['episodes'] = episodes
    return result


def get_show_stats(show, cache_dir):
    if not cache_dir:
        return infer_show_stats(show)

    name = show[0][1]
    year = show[0][2]
    filename = f"episodes-{hash_str_sha256([name, year])}.json"
    cached_data = read_cached_json(cache_dir, filename)
    if cached_data and 'expected' in cached_data:
        return cached_data['expected']

    try:
        imdb_id = imdb_get_movie_id(show, cache_dir)
        # print(f"INFO: imdb_id = {imdb_id}")

        cached_data = {
            'imdb_id': imdb_id,
            'name': name,
            'year': year,
        }
        if not imdb_id:
            cached_data['expected'] = infer_show_stats(show)
        else:
            movie = ia.get_movie(imdb_id, ('main', 'plot', 'episodes'))
            if movie and 'episodes' in movie:
                # Remove seasons less than 0. Not sure what "-1" indicates.
                cached_data['expected'] = {
                    'seasons': sorted(list(filter(lambda e: int(e) >= 0, movie['episodes'].keys()))),
                    'episodes': {},
                }
                for season_num in movie['episodes'].keys():
                    cached_data['expected']['episodes'][str(season_num)] = sorted(
                        list(filter(lambda e: int(e) > 0, movie['episodes'][season_num].keys())))
            else:
                cached_data['expected'] = infer_show_stats(show)

        write_cached_json(cache_dir, filename, cached_data)
        return cached_data['expected']
    except IMDbDataAccessError:
        # If expired cached data is available, return it
        cached_data = read_cached_json(cache_dir, filename, False)
        if cached_data and 'expected' in cached_data:
            return cached_data['expected']
        else:
            return infer_show_stats(show)


def completion(show, cache_dir):
    """
    Compute percent completion and missing seasons / episodes.
    The result is a tuple. Index 0 is an integer 0-100 of percent complete. Index 1 is a list of
    missing episodes in "S##E##" format, where an entire season is "S##".
    :param show:
    :param cache_dir:
    :return:
    """
    show_stats = get_show_stats(show, cache_dir)
    # print(f"Show stats: {show_stats}")
    existing = {}
    for e in show:
        existing.update({format_season(e[3]): True})
        existing.update({format_episode(e[3], e[4]): True})

    missing = []
    expected_total = 0
    existing_total = 0
    for s in show_stats['seasons']:
        s_str = format_season(s)
        if s_str not in existing:
            expected_total += len(show_stats['episodes'][str(s)])
            s_previous_str = format_season(s - 1)
            if s > 1 and len(missing) > 0 and missing[-1].endswith(s_previous_str):
                if missing[-1] == s_previous_str:
                    missing[-1] = s_previous_str + "-" + s_str
                else:
                    missing[-1] = missing[-1].replace(
                        s_previous_str, s_str)
            else:
                missing.append(s_str)
        else:
            for e in show_stats['episodes'][str(s)]:
                expected_total += 1
                e_str = format_episode(s, e)
                e_previous_str = format_episode(s, e - 1)
                if e_str in existing:
                    existing_total += 1
                else:
                    if e > 1 and len(missing) > 0 and missing[-1].endswith(e_previous_str):
                        if missing[-1] == e_previous_str:
                            missing[-1] = e_previous_str + "-" + e_str
                        else:
                            missing[-1] = missing[-1].replace(e_previous_str, e_str)
                    else:
                        missing.append(e_str)

    if expected_total == 0:
        completed = 0
    else:
        completed = int(100 * existing_total / expected_total)

    return completed, missing


def print_completion_csv(shows, file_no=sys.stdout, cache_dir=None):
    file_no.write('"Show","Section","Complete","Missing"\n')
    # require year
    for show in episode_human_sort(list(filter(lambda e: e[0][2], shows.values()))):
        c = completion(show, cache_dir)
        file_no.write(
            f'"{show[0][1]} ({show[0][2]})","{show[0][0]}","{c[0]}%","{",".join(c[1])}"\n')


def tvshow_summary_cli(argv) -> int:
    plex_url = common.get_plex_url()
    episode_csv = ''
    completion_csv = ''
    cache_dir = os.path.join(common.get_work_dir(), 'tvshow-summary')
    os.makedirs(cache_dir, exist_ok=True)

    try:
        opts, args = getopt.getopt(argv, "hu:e:c:t:", ["url=", "episodes=", "completion=", "cache_dir="])
    except getopt.GetoptError:
        usage()
        return 2
    for opt, arg in opts:
        if opt == '-h':
            usage()
            return 2
        elif opt in ("-u", "--url"):
            plex_url = arg
        elif opt in ("-e", "--episodes"):
            episode_csv = arg
        elif opt in ("-c", "--completion"):
            completion_csv = arg
        elif opt in ("-t", "--cache_dir"):
            cache_dir = arg

    shows = read_from_api(plex_url)

    if episode_csv:
        episode_fileno = open(episode_csv, "w")
    else:
        episode_fileno = sys.stdout
    print_episode_csv(shows, episode_fileno)
    if episode_csv:
        episode_fileno.close()

    if completion_csv:
        completion_fileno = open(completion_csv, "w")
    else:
        completion_fileno = sys.stdout
    print_completion_csv(shows, completion_fileno, cache_dir)
    if completion_csv:
        completion_fileno.close()

    return 0


if __name__ == '__main__':
    common.setup_cli()
    sys.exit(tvshow_summary_cli(sys.argv[1:]))
