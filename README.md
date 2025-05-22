# Media Hare

[![GitHub Issues](https://img.shields.io/github/issues-raw/double16/media-hare.svg)](https://github.com/double16/media-hare/issues)
[![Build](https://github.com/double16/media-hare/workflows/Build/badge.svg)](https://github.com/double16/media-hare/actions?query=workflow%3ABuild)
[![](https://img.shields.io/badge/Donate-Buy%20me%20a%20coffee-orange.svg)](https://www.buymeacoffee.com/patDj)

![](docs/media-hare.jpg)

Sundry tools for maintaining a personal media library. Rabbits (hares) like to be clean and are frequently grooming.
The tools here are intended to groom your media files for various purposes such as:

1. Transcode to storage optimized codecs
2. Cut commercials from DVR recordings
3. Profanity filtering
4. Be nice to low powered (compute, iops) machines
5. Tested with ATSC 1.0, DVD region 1 and Blu-Ray region 1. Should work with anything ffmpeg supports, but might need
   changes. Create an issue if you have problems. Include the output of `ffprobe` on the file.

## Work in Progress

This repository is a work in progress! I wrote it for my personal needs and thought others may benefit. Over time, I
want to make it configurable to apply to a wider audience. Pull requests are welcome! If you want to change something
like a codec or directory path, please make it configurable.

Assumptions:
1. Target codecs are H.265, H.264 (video) and Opus (audio). Text subtitles are SRT or ASS. Standard DVD and Blu-ray subtitles are kept.
2. Container is MKV. It supports multiple audio and subtitle streams and custom tags. This would be difficult to change. 
3. Dropbox is my "backup" solution. Code doesn't depend on that but some choices are made because of it. For example, temp files start with ".~" because Dropbox ignores them.
4. My media is served by Plex. Code generally does not depend on Plex, but supports interacting with it. I'm open to expanding support to other media servers.

The tools are intended to be run in a Docker container because there is specific software that needs to be installed
and configured. On some architectures it's difficult or may conflict with other software. So I use a Docker container.

If people want to support running outside of Docker, I'll take PRs for that, but I can't do much with support since I
won't be using it this way.

## TL;DR

I just want to run it ...

### media-hare.ini

Create a config file specific to your setup in your home directory as `.media-hare.ini` (notice the leading period
because POSIX people like that). You could also place in `/etc/media-hare.ini` if you'd like. You only need to add
options that are different from [media-hare.defaults.ini](dvrprocess/media-hare.defaults.ini). 

### Plex running on Docker

If you're running Plex on Docker, running is easier.

Consider running the (plex-hare)[https://github.com/double16/plex-hare] container. There are examples for using
Docker Compose. It integrates with `media-hare` and does process priority tuning.

Otherwise, run something similar to the following, replacing time zone and paths as necessary.

```shell
$ docker run -d -e "TZ=America/Chicago" --device /dev/dri --device /dev/nvidiactl --device /dev/nvidia0 --device /dev/nvidia-uvm -v /path/to/media:/media -v /path/to/media-hare.ini:/etc/media-hare.ini ghcr.io/double16/media-hare:main
```

### Plex running in anyway outside of Docker

Install Docker for Desktop from https://www.docker.com/products/docker-desktop/. Most of the defaults should be fine except that you likely want to increase the number of CPUs to 1 or 2 less than the maximum available. Transcoding is a compute heavy operation.

Open a shell like Terminal or Powershell and run the following, replacing time zone and paths as necessary.

```shell
$ docker run -d -e "TZ=America/Chicago" --device /dev/dri --device /dev/nvidiactl --device /dev/nvidia0 --device /dev/nvidia-uvm -v /path/to/media:/media -v /path/to/media-hare.ini:/etc/media-hare.ini ghcr.io/double16/media-hare:main
```

## Language Tool as Service

The LibreOffice language tool is used for spell checking and generating subtitles from audio. Only one is needed. The
two environment variables `LANGUAGE_TOOL_HOST` and `LANGUAGE_TOOL_PORT` are used to configure it.

```yaml
volumes:
   media_hare_pip:
   media_hare_python_packages:
   media_hare_whisper_cache:

services:

  langtool:
    image: ghcr.io/double16/libreoffice-langtool:main
    restart: unless-stopped
    ports:
      - "8100:8100/tcp"

  media-hare:
    image: ghcr.io/double16/media-hare:main
    restart: unless-stopped
    environment:
      - "TZ=America/Chicago"
      - "LANGUAGE_TOOL_HOST=langtool"
      - "LANGUAGE_TOOL_PORT=8100"
    volumes:
       - media_hare_pip:/var/cache/pip
       - media_hare_python_packages:/usr/local/lib/python3.12
       - media_hare_whisper_cache:/root/.cache/whisper
```

## Development Recommendation

You'll need to place `media-hare.ini` into your workspace directory.

- Mount media folder to /path/to/media
- Mount your source folder to ~/Workspace

### language tool server

Performance is better running the language tool on docker. Otherwise media-hare will it in each
process, using unnecessary memory and compute.

```shell
docker run -d --name langtool -p 8100:8100 ghcr.io/double16/libreoffice-langtool:main
export LANGUAGE_TOOL_HOST=localhost
export LANGUAGE_TOOL_PORT=8100
```

### docker / Docker Desktop

```shell
$ docker build -t media-hare:latest .

$ docker run -it --rm --entrypoint /bin/zsh --device /dev/dri --device /dev/nvidiactl --device /dev/nvidia0 --device /dev/nvidia-uvm -v ~/Movies:/Movies -v /path/to/media:/media -v .:/Workspace -v ~/.media-hare.ini:/etc/media-hare.ini media-hare:latest
```

### podman

```shell
$ podman machine init --cpus 10 --disk-size 30 -m 16384 -v ~/Movies:/Movies -v ~/Workspace:/Workspace -v /path/to/media:/media

$ podman build -t media-hare:latest .

$ podman run -it --rm --entrypoint /bin/zsh -v /Movies:/Movies -v /path/to/media:/media -v .:/Workspace /Workspace/media-hare.ini:/etc/media-hare.ini media-hare:latest
```

## Configuration

Configuration is in `media-hare.ini`. This can be found in `/etc/media-hare.ini` or `${HOME}/.media-hare.ini`. In your
media directories, you can also add `media-hare.ini` and override configurations for content in that directory and
subdirectories.

See `dvrprocess/media-hare.defaults.ini` for all available options and documentation. Edit `media-hare.ini` in one of the
paths specified above to override.

## dvr_post_process.py

Transcode videos to target codecs and other settings. Targets a specific language and keeps only one audio stream (it does preserve additional audio added by the profanity filter). Do not use if you want multiple language tracks.

See [docs/dvr_post_process.md](docs/dvr_post_process.md).

## find_need_transcode.py

Searches for media that needs to be transcoded. Supports several query parameters such as video and audio codec. Defaults to values in `media-hare.ini`.

## transcode-apply.py

Searches for media that needs to be transcoded and calls dvr_post_process.py. In a container runs periodically with time and size limits. See [docs/transcode-apply.md](docs/transcode-apply.md).

## profanity_filter.py

Filters profanity using subtitles. See [docs/profanity_filter.md](docs/profanity_filter.md).

## profanity-apply-filter.py

Searches for media that needs the profanity filter applied. In a container runs periodically with time and size limits. See [docs/profanity-apply-filter.md](docs/profanity-apply-filter.md).

## profanity-filter-report.py

Reports the changes made by the profanity filter in one or more mkv files. Use this to evaluate how your phrase lists are performing. See [docs/profanity-filter-report.md](docs/profanity-filter-report.md).

## comchap.py

Searches for commercials in media. Add chapters in media and EDL files for processing by other tools. See [docs/comchap.md](docs/comchap.md).

## comcut.py

Cuts commercials from media. See [docs/comcut.md](docs/comcut.md).

## comtune.py

Performs extra commercial skip tuning. For TV shows performs extensive tuning per season. See [docs/comtune.md](docs/comtune.md).

## smart-comcut.py

Intended to cut commercials for TV shows only when cuts look consistent and ignores outliers. See [docs/smart-comcut.md](docs/smart-comcut.md).

## scene-extract.py

Extracts scenes from media into separate files using an EDL file. See [docs/scene-extract.md](docs/scene-extract.md).

## find_need_comcut.py

Searches for media that needs commercials to be cut. Limited to paths with `Movies` in the name. TV shows could be a lot of results.
