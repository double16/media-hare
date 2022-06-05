# Media Hare

[![GitHub Issues](https://img.shields.io/github/issues-raw/double16/media-hare.svg)](https://github.com/double16/media-hare/issues)
[![Build](https://github.com/double16/media-hare/workflows/Build/badge.svg)](https://github.com/double16/media-hare/actions?query=workflow%3ABuild)
[![](https://img.shields.io/badge/Donate-Buy%20me%20a%20coffee-orange.svg)](https://www.buymeacoffee.com/patDj)

![](docs/media-hare.jpg)

Sundry tools for maintaining a personal media library. Rabbits (hares) like to be clean and are frequently grooming.
The tools here are intended to groom your media files for various purposes such as:

1. transcode to storage optimized codecs
2. cut commercials from DVR recordings
3. profanity filtering

## Work in Progress

This repository is a work in progress! I wrote it for my personal needs and thought others may benefit. Over time I
want to make it configurable to apply to a wider audience. Pull requests are welcome! If you want to change something
like a codec or directory path, please make it configurable.

Assumptions:
1. Target codecs are H.264 (video) and Opus (audio). Text subtitles are SRT or ASS. Standard DVD and Blu-ray subtitles are kept.
2. Dropbox is my "backup" solution, so base directory is /home/Dropbox and temp files are ".~" because Dropbox ignores them.
3. My media is served by Plex. Code generally does not depend on Plex, but supports interacting with it.

The tools are intended to be run in a Docker container because there is specific software that needs to be installed
and configured. On some architectures it's difficult or may conflict with other software. So I use a Docker container.

The following will run the tools periodically and limit runs by size of content changed and run time.

```shell
$ docker run -d -e "TZ=America/Chicago" -v /path/to/media:/home/Dropbox ghcr.io/double16/media-hare:main
```

## Development Recommendation

### docker / Docker Desktop

```shell
$ docker build -t media-hare:latest .

$ docker run -it --rm --entrypoint /bin/zsh -v ~/Movies:/Movies -v ~/Mounts/dropbox:/home/Dropbox -v .:/Workspace media-hare:latest
```

### podman

- Mount media folder to ~/Mounts/dropbox
- Mount your source folder to ~/Workspace

```shell
$ podman machine init --cpus 10 --disk-size 30 -m 16384 -v ~/Movies:/Movies -v ~/Workspace:/Workspace -v ~/Mounts/dropbox:/home/Dropbox

$ podman build -t media-hare:latest .

$ podman run -it --rm --entrypoint /bin/zsh -v /Movies:/Movies -v /home/Dropbox:/home/Dropbox -v .:/Workspace media-hare:latest
```

## dvr_post_process.py

TODO

## transcode-apply.py

TODO

## profanity_filter.py

TODO

## profanity-apply-filter.py

TODO

## comchap.py

TODO

## comcut.py

TODO

## comtune.py

TODO

## smart-comcut.py

TODO

## ffmpeg recipes

Trim (this can have dead space at the beginning if not on an I-FRM boundary):

```shell
$ ffmpeg -ss 00:00:18 -i x.mkv -to 01:05:20 -c copy y.mkv
```

Shift subtitle:

```shell
$ ffmpeg -itsoffset 1.5 -i x.srt -c copy y.srt
```

Add subtitle

```shell
$ ffmpeg -i x.mkv -i x.srt -map 0 -map 1 -c copy -metadata:s:a language=eng -metadata:s:s language=eng y.mkv
```
