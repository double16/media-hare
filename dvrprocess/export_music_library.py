#!/usr/bin/env python3

import json
from plexapi.server import PlexServer
import argparse

import common


def get_music_library(plex, token=None):
    music_data = []

    # Fetch music library
    music_section = plex.library.section('Music')

    for album in music_section.albums():
        artist_name = album.artist().title
        album_title = album.title
        album_year = album.year

        for track in album.tracks():
            track_data = {
                "artist": artist_name,
                "album": album_title,
                "year": album_year,
                "title": track.title
            }
            music_data.append(track_data)

    return music_data

def main():
    parser = argparse.ArgumentParser(description="Extract music library from Plex Media Server.")
    parser.add_argument("--token", type=str, help="Optional Plex API token")
    parser.add_argument("--filename", type=str, default="music_library.json", help="Output JSON filename")
    parser.add_argument("--url", type=str, default=common.get_plex_url(), help="Base URL of the Plex Media Server")
    args = parser.parse_args()

    plex = PlexServer(args.url, args.token)

    music_data = get_music_library(plex, args.token)

    # Write to the specified JSON file
    with open(args.filename, 'w') as f:
        json.dump(music_data, f, indent=4)

    print(f"Music library exported to {args.filename}")

if __name__ == "__main__":
    main()
