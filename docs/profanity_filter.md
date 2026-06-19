# profanity_filter.py

The profanity filter uses text based subtitles to mute audio and mask subtitles for offensive words and phrases. Media must have either an SRT or ASS subtitle. If one of those isn't available but a DVD vobsub or Blu-Ray hdmv_pgs subtitle is present (these are the standard formats), an attempt will be made to convert to text using the (Subtitle-Edit)[https://github.com/SubtitleEdit/subtitleedit] program. If no image based subtitle is available, audio to text is attempted using Whisper. If at least 93% of words are found in the dictionary (using the hunspell library), the text subtitle is used for filtering.

The filter output produces several streams. The original audio and subtitle streams are kept. A filtered audio stream is added. Two filtered subtitle streams are added. One is all subtitles with filtering. The second only contains filtered frames and is marked as 'forced'. Using this stream with the filtered audio will only show subtitles when filtering is applied.

The filter also adds tags to the media so that changes to the phrase list can be applied. Re-running the filter will only change the media if changes to filtering are found. If the phrase list has been changed such that no filtering is done, the filtered streams will be removed from the media.

The phrase list is divided into a censor list and stop list. The censor list only masks the phrases in the list and keeps other text. The stop list masks the entire frame of text. The intent of the stop list is to identify undesirable subject matter such that only masking a word would still present offensive content.  There is an allow list for special cases for acceptable phrases that match something in the censor list.

The determination of "offensive" is highly subjective. There is an issue to allow the censor and stop list to be stored externally so that each install can tailor the lists to personal preference. Possibly allowing multiple lists will various levels (such as aligning with the U.S. designations of G, PG, PG-13, R, etc.)

Filtering can be done after post-processing in the [dvrprocess/dvr_post_process.py] program with either the `--profanity-filter` option or configuring `post_process.profanity_filter=true` in `media-hare.ini`.

The `profanity-filter-apply.py` program is run regularly to find media that needs filtering applied.

The script has a `--mark-skip` option to mark file(s) for skipping the filter. There is an `--unmark-skip` to reverse that.

## Subtitle Audio Alignment

Correct subtitle timing matters because the filter uses subtitle text to decide which audio spans to silence. The preferred text source is still the original subtitle track when it is usable: human-authored subtitles normally have better wording, punctuation, speaker cues, and segmentation than an automatic transcript. Whisper is used as a timing and gap-filling source, not as the primary subtitle replacement.

The active alignment implementation is `fix_subtitle_audio_alignment()` in `dvrprocess/profanity_filter.py`. It aligns original subtitle events to Whisper word-level transcript entries with a monotonic dynamic-programming pass:

1. Normalize subtitle and transcript text into comparable word tokens while keeping the original text for output.
2. Build several candidate transcript word ranges for each subtitle event. Candidates come from the expected timing window, nearby transcript windows, fuzzy text matches, and partial phrase matches inside longer transcript runs.
3. Score each candidate with text similarity, token coverage, word-count fit, duration fit, and proximity to the original subtitle timing. Interior phrase matches are allowed so that a subtitle can align to the relevant words even when Whisper includes extra surrounding speech.
4. Use dynamic programming to choose one candidate per subtitle event, or skip alignment for that event, while enforcing monotonic word order. A later subtitle cannot claim transcript words that occur before or overlap words claimed by an earlier subtitle.
5. Apply the selected matches to retime original subtitle events. Unmatched events are interpolated conservatively between neighboring aligned events so their relative order is preserved.
6. Add short transcript-only gaps as new subtitle events when Whisper contains words or phrases that were missing from the original subtitle. Insertions are intentionally conservative because automatic transcripts can contain hallucinations and duplicate fragments.
7. Normalize the final event list so events have positive duration, remain sorted, and do not overlap.

The most important invariant is monotonicity. The alignment may choose a lower-scoring local match when that preserves the global subtitle order; this is preferable to retiming one subtitle onto an earlier repeated phrase and shifting later profanity masking onto the wrong audio. Repeated phrases, dropped subtitle words, and extra transcript words should be handled by the global order constraint rather than by local best-match decisions.

## Alignment Maintenance Notes

When maintaining the alignment code, tune scoring and candidate generation before adding special-case post-processing. The main maintainability boundary is:

- Candidate data is represented by `SubtitleAlignmentCandidate`.
- Dynamic-programming state is represented by `SubtitleAlignmentState`.
- Token normalization for transcript words is handled by `_transcribed_word_to_plain_tokens()`.
- The public behavior is in `fix_subtitle_audio_alignment()`. The legacy implementation is retained privately for reference only and should not receive behavior changes unless it is intentionally restored.

Prefer preserving original subtitle text unless the transcript exposes a missing word range between existing subtitles. Missing transcript-only ranges should be inserted only when they fit into a real timing gap and do not duplicate adjacent subtitle text. This keeps the human subtitle track authoritative while still allowing the filter to catch profanity that the original subtitle omitted.

Tests should protect behavior instead of exact fixture output where possible. Useful regression checks include:

- aligned subtitles remain sorted, positive-duration, and non-overlapping;
- repeated phrases align to transcript words in subtitle order;
- extra transcript words can be inserted without shifting surrounding original subtitles;
- homophones and spelling corrections do not replace the original human subtitle wording;
- fixture episodes still produce a material improvement without requiring byte-for-byte timing matches.

The subtitle alignment tests use the LanguageTool server when available. For local runs against an already-started server, use:

```shell
LANGUAGE_TOOL_PORT=8100 ../.venv/bin/python -m pytest test_subtitle_alignment.py -q
```
