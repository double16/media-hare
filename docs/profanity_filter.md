# profanity_filter.py

The profanity filter uses text based subtitles to mute audio and mask subtitles for offensive words and phrases. Media must have either an SRT or ASS subtitle. If one of those isn't available but a DVD vobsub or Blu-Ray hdmv_pgs subtitle is present (these are the standard formats), an attempt will be made to convert to text using the (Subtitle-Edit)[https://github.com/SubtitleEdit/subtitleedit] program. If no image based subtitle is available, audio to text is attempted using (Vosk)[https://alphacephei.com/vosk]. If at least 93% of words are found in the dictionary (using the hunspell library), the text subtitle is used for filtering.

The filter output produces several streams. The original audio and subtitle streams are kept. A filtered audio stream is added. Two filtered subtitle streams are added. One is all subtitles with filtering. The second only contains filtered frames and is marked as 'forced'. Using this stream with the filtered audio will only show subtitles when filtering is applied.

The filter also adds tags to the media so that changes to the phrase list can be applied. Re-running the filter will only change the media if changes to filtering are found. If the phrase list has been changed such that no filtering is done, the filtered streams will be removed from the media.

The phrase list is divided into a censor list and stop list. The censor list only masks the phrases in the list and keeps other text. The stop list masks the entire frame of text. The intent of the stop list is to identify undesirable subject matter such that only masking a word would still present offensive content.  There is an allow list for special cases for acceptable phrases that match something in the censor list.

The determination of "offensive" is highly subjective. There is an issue to allow the censor and stop list to be stored externally so that each install can tailor the lists to personal preference. Possibly allowing multiple lists will various levels (such as aligning with the U.S. designations of G, PG, PG-13, R, etc.)

Filtering can be done after post-processing in the [dvrprocess/dvr_post_process.py] program with either the `--profanity-filter` option or configuring `post_process.profanity_filter=true` in `media-hare.ini`.

The `profanity-filter-apply.py` program is run regularly to find media that needs filtering applied.

The script has a `--mark-skip` option to mark file(s) for skipping the filter. There is an `--unmark-skip` to reverse that.
