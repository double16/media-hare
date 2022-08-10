# EDL

The Edit Description List (EDL) file format is used to specify video cut and mute points. Some players can process these  in line. For those that cannot, there are several tools here that modify a media file with an EDL file. The following is the EDL format this tool set uses:

```
## start        end             type    file
100.2           360.0           0
00:01:05.456    00:02:10.000    1
#32750          #40000          3
00:01:00.000    00:05:00.000    2       Show - S01E01
00:05:10.000    00:05:30.000    4
```

The start and end fields can be one of:
- Seconds with decimal, adjusted to I-frame unless video transcoding
- HH:MM:SS.SSS, adjusted to I-frame unless video transcoding
- #12345 - Frame number

The type field is an integer of one of the following values:
- 0 - Cut the section
- 1 - Mute the section
- 2 - Scene marker, `scene-extract.py` will extract these into separate files and name them as in the `file` column
- 3 - Commercial section, considered the same as 0, i.e. cut it.
- 4 - Blur the video background

The file column is optional and is only used by `scene-extract.py`. If specified, the scene will use this file name. Otherwise the source file name plus a number will be used.
