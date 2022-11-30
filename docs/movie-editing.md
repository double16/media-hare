# Movie Editing

My process for editing movies for content and language follows. It's best to edit the movie using the native rip from
the media (DVD, Blu-Ray, DVR, ...). Editing video that's been transcoded for storage will have fewer key frames and
likely require re-encoding to cut. This process will preserve subtitles, I have yet to find software that doesn't lose
subtitles or mangle them or is very difficult to use.

1. Use MakeMKV to rip. Choose the audio for your language with the highest bitrate. Choose the first subtitle, don't pick any forced subtitle. If a subtitle with "ccextractor" is available, also pick that.
2. Run `profanity_filter.py` on the `.mkv` file. Inspect the tags using `ffprobe` and look for `PFILTER_STOPPED`. These are indicators of possible scenes to cut. Also run `profanity-filter-report.py` on the `.mkv` file to see the list of filtered subtitles.
3. Review the "Parent's Guide" on imdb.com for the movie for scenes you may want to cut.
4. Open a `.edl` file using the same name of the movie, replacing `.mkv` with `.edl`.
5. Open in avidemux to find cut and mute points and add to the EDL file. Don't cut with avidemux. Each line must have start time, end time and an integer indicating what you want to do. You can copy and paste from the time marker in avidemux.
    ```
    # Cut the scene:
    00:02:30.425    00:02:35.500    0
    # Mute the audio and mask the subtitle:
    00:02:30.425    00:02:35.500    1
    # Blur the background
    00:02:30.425    00:02:35.500    3
    ```
6. Use `comcut.py` to cut the file. You can give the program the input file and an output file so you can review the cuts without modifying the original.
7. Use `dvr_post_process.py` to compress for storage.
