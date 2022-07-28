# ffmpeg recipes

Trim (this can have dead space at the beginning if not on an I-FRM boundary). Try to use `comcut.py`, it will align
to I-FRM and uses the concat filter for efficient multiple cuts.

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
