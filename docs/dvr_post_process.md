# dvr_post_process.py

This program transcodes your video files into desired codecs and other settings. It has several filters it can apply
in addition to transcoding. It is safe to re-run on already processed files. It will detect if everything is as
desired and not change the file. It will also only change things that do not match the desired settings. If you change
configuration you can re-run the program and fix up your content.

## .dvrconfig

The global settings are in `media-hare.ini`. However, directory (and it's subdirectories) specific settings can be
set in a `.dvrconfig` file. This file contains arguments to `dvr_post_process.py` as you would pass them on the
command line. One option per line, including arguments.

```
--stereo
--height=480
--crop-frame
```

This file will mix down audio to stereo, scale the height to 480 and preserve aspect ratio, and also crop black bars
around the video.
