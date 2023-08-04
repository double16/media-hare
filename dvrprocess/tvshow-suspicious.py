#!/usr/bin/env python3

import getopt
import logging
import sys
import xml.etree.ElementTree as ET
import requests
import common

#
# Find episodes that may have been recorded or cut incorrectly.
#


def usage():
    print(f"""
Find episodes that may have been recorded or cut incorrectly.

Usage: {sys.argv[0]} -u http://127.0.0.1:32400 -o episodes-suspicious.csv

-u, --url=
    The Plex Media Server URL. Specify as http://127.0.0.1:32400, default is {common.get_plex_url()}
-o, --output=
    Location of the CSV output file. If not specified, stdout.
""", file=sys.stderr)


def read_from_api(plex_url, limit=None):
    result = {}
    # set of tuple of (title, season, episode)
    episodes_by_show = set() 
    sections_response = requests.get(f'{plex_url}/library/sections')
    sections = list(
        filter(lambda el: el.tag == 'Directory' and el.attrib['type'] == 'show',
               ET.fromstring(sections_response.text)))

    for section in sections:
        section_response = requests.get(
            f'{plex_url}/library/sections/{section.attrib["key"]}/all')
        shows = list(filter(lambda el: el.tag ==
                                       'Directory' and el.attrib['type'] == 'show',
                            ET.fromstring(section_response.text)))
        for show in shows:
            show_all = ET.fromstring(requests.get(f'{plex_url}{show.attrib["key"]}'.replace('/children', '')).text)[0]
            genres = list(map(lambda el: el.attrib['tag'], filter(lambda el: el.tag == 'Genre', show_all)))
            if 'Short' in genres:
                continue
            show_response = requests.get(
                f'{plex_url}{show.attrib["key"].replace("/children", "/allLeaves")}')
            episodes = list(filter(
                lambda el: el.tag == 'Video' and el.attrib['type'] == 'episode' and 'index' in el.attrib,
                ET.fromstring(show_response.text)))
            episode_list = []
            for episode in episodes:
                episode_num = int(episode.attrib["index"])
                season = int(episode.attrib["parentIndex"])

                # season 0 are specials, hard to tell what's suspicious
                if season == 0:
                    continue

                duration = episode.attrib.get("duration")
                if duration is None:
                    continue

                # check for suspicious duration
                duration = int(int(duration) / 60000)
                if duration >= 119:
                    # episodes over 120 are usually specials, hard to tell what's suspicious
                    continue

                if duration >= 18:
                    continue

                key = (show.attrib["title"], season, episode_num)
                if key in episodes_by_show:
                    continue
                episodes_by_show.add(key)
    
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
                result[section.attrib["title"]+":"+show.attrib["title"]] = episode_list
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


def tvshow_suspicious_cli(argv) -> int:
    plex_url = common.get_plex_url()
    episode_csv = ''

    try:
        opts, args = getopt.getopt(argv, "hu:e:c:t:", ["url=", "output="])
    except getopt.GetoptError:
        usage()
        return 2
    for opt, arg in opts:
        if opt == '-h':
            usage()
            return 2
        elif opt in ("-u", "--url"):
            plex_url = arg
        elif opt in ("-e", "--output"):
            episode_csv = arg

    shows = read_from_api(plex_url)

    if episode_csv:
        episode_fileno = open(episode_csv, "w")
    else:
        episode_fileno = sys.stdout
    print_episode_csv(shows, episode_fileno)
    if episode_csv:
        episode_fileno.close()

    return 0


if __name__ == '__main__':
    common.setup_cli(level=logging.ERROR)
    sys.exit(tvshow_suspicious_cli(sys.argv[1:]))
