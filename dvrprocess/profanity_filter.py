#!/usr/bin/env python3

import atexit
import copy
import getopt
import hashlib
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from bisect import bisect_left, bisect_right

import whisper
from math import ceil, floor
from pathlib import Path
from statistics import mean, stdev
from typing import Tuple, Union

import language_tool_python
import pysrt
from ass_parser import read_ass, write_ass, AssFile, AssEventList, CorruptAssLineError
from num2words import num2words
from numpy import loadtxt, average, concatenate
from pysrt import SubRipItem, SubRipFile, SubRipTime
from thefuzz import fuzz
from thefuzz import process as fuzzprocess
from whisper.model import Whisper

import common
from common import subtitle, tools, config, constants, progress, edl_util, fsutil
from common.proc_invoker import StreamCapturingProgress

# Increment when a coding change materially effects the output
FILTER_VERSION = 12
AUDIO_TO_TEXT_VERSION = 7
AUDIO_TO_TEXT_SUBTITLE_VERSION = 6

# exit code for content had filtering applied, file has been significantly changed
CMD_RESULT_FILTERED = 0
# exit code for content version mark has been updated, file has been trivially changed
CMD_RESULT_MARKED = 1
# exit code for content is unchanged
CMD_RESULT_UNCHANGED = 2
# exit code for general error
CMD_RESULT_ERROR = 255

# When creating text from bitmaps or audio, what is the minimum percentage of dictionary words we require?
WORD_FOUND_PCT_THRESHOLD = 93.0

# Number of milliseconds between words to assume a new sentence.
SILENCE_FOR_NEW_SENTENCE = 1200
# Number of milliseconds between words that may fit a sound effect.
SILENCE_FOR_SOUND_EFFECT = 500

ASSA_TYPEFACE_REMOVE = re.compile(r"[{][\\][iubsIUBS]\d+[}]")

WHISPER_MODEL_NAME_FALLBACK = "medium"
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", WHISPER_MODEL_NAME_FALLBACK)
__WHISPER_MODELS: list[Whisper] = list()
WHISPER_BEAM_SIZE = 5
WHISPER_PATIENCE = 2.5

logger = logging.getLogger(__name__)

debug = False


def usage():
    print(f"""{sys.argv[0]}

Filter audio and subtitles for profanity.
- The original audio and subtitles are kept titled 'Original'. Filtered streams are titled 'Filtered' and are the
  default streams.
- The default audio and subtitle streams are used for filtering. Any additional streams are preserved.
- The script may be run multiple times on the same video and the file will only be changed if the result changes.
- Filtered streams will be removed if changes occur such that filtering is no longer needed.
- The 'censor_list.txt' file contains phrases that are muted and the phrase is marked out in the subtitle.
- The 'stop_list.txt' file contains phrases that are muted and the entire subtitle is marked.
- The 'allow_list.txt' file contains phrases that match `censor_list.txt` but should be allowed.
- Use --mark-skip to skip filtering, or set the tag 'PFILTER_SKIP' in the video to 'y' or 't'.

--dry-run
    Output command that would be used but do nothing
--mute-voice-channels
    Mute only voice channels for audio with 3+ channels
--mute-all-channels
    Mute all channels for audio with 3+ channels
--keep
    Keep original file in a backup prefixed by ".~"
--debug
    Only create hidden output file for debugging
--work-dir={config.get_work_dir()}
--remove
    Remove filtering
--mark-skip
    Mark file(s) for the filter to skip. If a file has been filtered, filtering will be removed. 
--unmark-skip
    Unmark file(s) for the filter to skip, so they will be filtered. Filtering will be performed. 
--force
    Force re-filtering
    
Environment:
    LANGUAGE_TOOL_HOST=127.0.0.1
    LANGUAGE_TOOL_PORT=8100
""", file=sys.stderr)


def lazy_get_whisper_models() -> list[Whisper]:
    global __WHISPER_MODELS
    if not __WHISPER_MODELS:
        logger.info(f"Loading whisper model {WHISPER_MODEL_NAME}")
        __WHISPER_MODELS.append(whisper.load_model(WHISPER_MODEL_NAME))
        # fall back to medium because sometimes large returns no transcription
        if WHISPER_MODEL_NAME != WHISPER_MODEL_NAME_FALLBACK:
            logger.info(f"Loading fall back whisper model {WHISPER_MODEL_NAME_FALLBACK}")
            __WHISPER_MODELS.append(whisper.load_model(WHISPER_MODEL_NAME_FALLBACK))
    return __WHISPER_MODELS


def profanity_filter(*args, **kwargs) -> int:
    final_result = 0
    for input_file in list(args):
        try:
            final_result = max(do_profanity_filter(input_file, **kwargs), final_result)
        except CorruptAssLineError:
            logger.error("Corrupt ASS subtitle in %s", args[0])
            final_result = CMD_RESULT_ERROR
    return final_result

def do_profanity_filter(input_file, dry_run=False, keep=False, force=False, filter_skip=None, mark_skip=None,
                        unmark_skip=None, language=constants.LANGUAGE_ENGLISH, workdir=None, verbose=False,
                        mute_channels: Union[None, config.MuteChannels] = None) -> int:
    if mark_skip and unmark_skip:
        logger.fatal("mark-skip and unmark-skip both set")
        return CMD_RESULT_ERROR

    if not os.path.isfile(input_file):
        logger.fatal(f"{input_file} does not exist")
        return CMD_RESULT_ERROR

    filename = os.path.realpath(input_file)
    base_filename = os.path.basename(filename)
    input_type = base_filename.split(".")[-1]
    dir_filename = os.path.dirname(filename)
    if workdir is None:
        workdir = config.get_work_dir()

    logger.info(f"filtering {filename}")

    # Temporary File Name for transcoding, we want to keep on the same filesystem as the input
    temp_base = os.path.join(workdir, f".~{'.'.join(base_filename.split('.')[0:-1])}")
    debug_base = os.path.join(dir_filename, f"{'.'.join(base_filename.split('.')[0:-1])}")
    temp_filename = os.path.join(dir_filename,
                                 f".~{'.'.join(base_filename.split('.')[0:-1])}.transcoded.{input_type}")
    # Hides filename from user UI and Dropbox
    hidden_filename = f"{dir_filename}/.~{base_filename}"
    if debug:
        output_filename = os.path.join(dir_filename,
                                       f"{'.'.join(base_filename.split('.')[0:-1])}.filtered.{input_type}")
    else:
        output_filename = os.path.join(dir_filename, f"{'.'.join(base_filename.split('.')[0:-1])}.{input_type}")

    if input_type != 'mkv':
        logger.fatal("Only MKV is supported")
        return CMD_RESULT_ERROR

    if common.assert_not_transcoding(input_file, exit=False) != 0:
        return CMD_RESULT_ERROR

    os.nice(12)

    # A limited set of codecs appear to be in the Plex Transcoder 1.19.5
    input_info = common.find_input_info(filename)

    if dry_run:
        logger.info(f"{input_info}")

    if mark_skip:
        filter_skip = True
    if filter_skip is None and not unmark_skip:
        filter_skip = common.is_truthy(
            input_info.get(constants.K_FORMAT, {}).get(constants.K_TAGS, {}).get(constants.K_FILTER_SKIP))

    if mute_channels is None:
        mute_channels_tag = input_info.get(constants.K_FORMAT, {}).get(constants.K_TAGS, {}).get(
            constants.K_MUTE_CHANNELS)
        if mute_channels_tag is not None:
            mute_channels = config.mute_channels(mute_channels_tag)

    streams_file = 0

    #
    # Construct command
    #

    # Global arguments
    arguments = []
    arguments.extend(['-hide_banner', '-y', '-analyzeduration', common.ANALYZE_DURATION,
                      '-probesize', common.PROBE_SIZE])

    arguments.extend(["-i", filename])

    subtitle_extract_command = arguments.copy()

    # Load word lists
    censor_list = load_censor_list()
    stop_list = load_stop_list()
    allow_list = load_allow_list()

    # Compute filter hash
    filter_hash = compute_filter_hash(censor_list, stop_list, allow_list)

    # Compare and exit if the same
    current_filter_version = input_info.get(constants.K_FORMAT, {}).get(constants.K_TAGS, {}).get(
        constants.K_FILTER_VERSION)
    current_filter_hash = input_info.get(constants.K_FORMAT, {}).get(constants.K_TAGS, {}).get(constants.K_FILTER_HASH)
    current_audio2text_version = input_info.get(constants.K_FORMAT, {}).get(constants.K_TAGS, {}).get(
        constants.K_AUDIO_TO_TEXT_VERSION)
    current_audio2text_subtitle_version = input_info.get(constants.K_FORMAT, {}).get(constants.K_TAGS, {}).get(
        constants.K_AUDIO_TO_TEXT_SUBTITLE_VERSION)
    transcribe_notes = input_info.get(constants.K_FORMAT, {}).get(constants.K_TAGS, {}).get(
        constants.K_AUDIO_TO_TEXT_NOTES)
    logger.info("current filter hash = %s, current filter version = %s, current audio-to-text version = %s",
                current_filter_hash, current_filter_version, current_audio2text_version)

    if not force:
        if current_filter_version and int(current_filter_version) > FILTER_VERSION:
            logger.info("Future filter version found: %s", current_filter_version)
            return CMD_RESULT_UNCHANGED
        if current_audio2text_version and int(current_audio2text_version) > AUDIO_TO_TEXT_VERSION:
            logger.info("Future audio2text version found: %s", current_audio2text_version)
            return CMD_RESULT_UNCHANGED
        if current_audio2text_subtitle_version and int(
                current_audio2text_subtitle_version) > AUDIO_TO_TEXT_SUBTITLE_VERSION:
            logger.info("Future audio2text subtitle version found: %s", current_audio2text_subtitle_version)
            return CMD_RESULT_UNCHANGED

    # Find original and filtered subtitle
    subtitle_original, subtitle_filtered, subtitle_filtered_forced, subtitle_words = common.find_original_and_filtered_streams(
        input_info,
        constants.CODEC_SUBTITLE,
        constants.CODEC_SUBTITLE_TEXT_BASED,
        language)
    # Find original and filtered audio stream
    audio_original, audio_filtered, _, _ = common.find_original_and_filtered_streams(input_info,
                                                                                     constants.CODEC_AUDIO,
                                                                                     None,
                                                                                     language)
    tags_filename = f"{temp_base}.tags.xml"
    if not debug:
        common.TEMPFILENAMES.append(tags_filename)

    if filter_skip:
        if audio_filtered is None and subtitle_filtered is None:
            logger.info("%s: filter skipped due to %s property", filename, constants.K_FILTER_SKIP)
            if mark_skip:
                return _tag_as_skipped(filename, tags_filename, input_info, dry_run=dry_run, debug=debug,
                                       verbose=verbose)
            return CMD_RESULT_UNCHANGED
        else:
            logger.info("%s: removing filter due to %s property", filename, constants.K_FILTER_SKIP)
    else:
        if not force and current_filter_hash == filter_hash and current_filter_version == str(
                FILTER_VERSION) and current_audio2text_version in [None, '',
                                                                   str(AUDIO_TO_TEXT_VERSION)] and current_audio2text_subtitle_version in [
            None, '', str(AUDIO_TO_TEXT_SUBTITLE_VERSION)]:
            logger.info("Stream is already filtered")
            return CMD_RESULT_UNCHANGED

    if not subtitle_original and filter_skip:
        if mark_skip:
            return _tag_as_skipped(filename, tags_filename, input_info, dry_run=dry_run, debug=debug, verbose=verbose)
        return CMD_RESULT_UNCHANGED

    media_duration = float(input_info[constants.K_FORMAT][constants.K_DURATION])
    subtitle_srt_generated = None
    subtitle_srt_words = None
    audio_to_text_version = current_audio2text_version
    if subtitle_original and constants.K_TAGS in subtitle_original:
        audio_to_text_subtitle_version = subtitle_original[constants.K_TAGS].get(
            constants.K_AUDIO_TO_TEXT_SUBTITLE_VERSION, '') or subtitle_original[constants.K_TAGS].get(
            constants.K_AUDIO_TO_TEXT_VERSION, '')
    else:
        audio_to_text_subtitle_version = ''
    logger.info("current audio-to-text version = %s", audio_to_text_subtitle_version)
    audio_to_text_filter = (subtitle_original or {}).get(constants.K_TAGS, {}).get(constants.K_AUDIO_TO_TEXT_FILTER, '')

    if need_original_subtitle_ocr(subtitle_original=subtitle_original,
                                  media_duration=media_duration,
                                  force=force):
        subtitle_srt_generated = ocr_subtitle_bitmap_to_srt(input_info, temp_base, language, verbose=verbose)

    if audio_original and (not filter_skip or current_audio2text_subtitle_version):
        if detect_transcribed_by_version_3(current_audio2text_version, input_info, subtitle_original):
            subtitle_original[constants.K_TAGS][constants.K_AUDIO_TO_TEXT_VERSION] = "3"
            audio_to_text_subtitle_version = "3"

        if need_words_transcribed(subtitle_words, audio_to_text_version, force):
            logger.info("%s Transcribing for words", base_filename)
            audio_channels = int(audio_original.get(constants.K_CHANNELS, 0))
            # audio_to_text_filter = "acompressor=threshold=-30dB:ratio=3:attack=10:release=200:knee=6:makeup=6,loudnorm=I=-16:TP=-1.5"
            audio_to_text_filter = "loudnorm=I=-16:TP=-1.5"
            if audio_channels > 2:
                # audio_to_text_filter = 'pan=stereo|FL<FL+FC|FR<FR+FC,'+audio_to_text_filter
                audio_to_text_filter = 'pan=mono|FC<FC+0.5*FL+0.5*FR,' + audio_to_text_filter
            subtitle_srt_words, transcribe_notes = audio_to_words_srt(input_info, audio_original, workdir,
                                                                      audio_to_text_filter, language)
            audio_to_text_version = AUDIO_TO_TEXT_VERSION
            if subtitle_srt_words and os.stat(subtitle_srt_words).st_size == 0:
                subtitle_srt_words = None

        if subtitle_srt_generated is None and (
                subtitle_srt_words or subtitle_words) and need_original_subtitle_transcribed(subtitle_original,
                                                                                             audio_to_text_subtitle_version,
                                                                                             media_duration, force):
            logger.info("%s Generating original subtitle from transcription", base_filename)
            subtitle_original_words_filename = None
            if not subtitle_srt_words:
                subtitle_original_words_filename = f"{temp_base}.original.words.srt"
                common.TEMPFILENAMES.append(subtitle_original_words_filename)
                args = ['-nostdin', "-loglevel", "error",
                        '-i', input_info['format']['filename'],
                        '-map', f'0:{subtitle_words[constants.K_STREAM_INDEX]}',
                        '-c', 'copy',
                        '-f', 'srt', subtitle_original_words_filename]
                tools.ffmpeg.run(args, check=True)
            _srt_text = words_to_subtitle_srt(input_info, subtitle_srt_words or subtitle_original_words_filename,
                                              workdir, language)
            if _srt_text and os.stat(_srt_text).st_size > 0:
                audio_to_text_subtitle_version = AUDIO_TO_TEXT_SUBTITLE_VERSION
                subtitle_srt_generated = _srt_text

    tags = input_info[constants.K_FORMAT].get(constants.K_TAGS, {}).copy()
    tags[constants.K_FILTER_HASH] = filter_hash
    tags[constants.K_FILTER_VERSION] = FILTER_VERSION
    tags[constants.K_AUDIO_TO_TEXT_VERSION] = audio_to_text_version if audio_to_text_version else ''
    tags[constants.K_AUDIO_TO_TEXT_FILTER] = audio_to_text_filter if audio_to_text_filter else ''
    tags[
        constants.K_AUDIO_TO_TEXT_SUBTITLE_VERSION] = audio_to_text_subtitle_version if audio_to_text_subtitle_version else ''
    tags[
        constants.K_AUDIO_TO_TEXT_NOTES] = transcribe_notes if transcribe_notes else ''
    if mute_channels:
        tags[constants.K_MUTE_CHANNELS] = mute_channels.name
    if common.should_replace_media_title(input_info):
        tags[constants.K_MEDIA_TITLE] = common.get_media_title_from_filename(input_info)
    tags[constants.K_MEDIA_PROCESSOR] = constants.V_MEDIA_PROCESSOR

    if not subtitle_original and subtitle_srt_generated is None:
        # mark as filtered so we don't repeat
        logger.fatal("Cannot find text based subtitle")
        common.write_mkv_tags(tags, tags_filename)
        if not dry_run and not debug:
            tools.mkvpropedit.run([filename, "--tags", f"global:{tags_filename}"])
        return CMD_RESULT_UNCHANGED

    if not audio_original:
        if filter_skip:
            if mark_skip:
                return _tag_as_skipped(filename, tags_filename, input_info, dry_run=dry_run, debug=debug,
                                       verbose=verbose)
            return CMD_RESULT_UNCHANGED
        else:
            # mark as filtered so we don't repeat
            logger.fatal("Cannot find audio stream for original")
            common.write_mkv_tags(tags, tags_filename)
            if not dry_run and not debug:
                tools.mkvpropedit.run([filename, "--tags", f"global:{tags_filename}"])
            return CMD_RESULT_UNCHANGED

    if subtitle_srt_generated is not None:
        subtitle_codec = constants.CODEC_SUBTITLE_SRT
    else:
        subtitle_codec = subtitle_original.get(constants.K_CODEC_NAME)
        if subtitle_codec == constants.CODEC_SUBTITLE_SUBRIP:
            subtitle_codec = constants.CODEC_SUBTITLE_SRT
    subtitle_original_filename = f"{temp_base}.original.{subtitle_codec}"
    subtitle_filtered_filename = f"{temp_base}.filtered.{subtitle_codec}"
    subtitle_filtered_forced_filename = f"{temp_base}.filtered-forced.{subtitle_codec}"
    subtitle_filtered_previous_filename = f"{temp_base}.filtered.previous.{subtitle_codec}"
    subtitle_words_filename = f"{temp_base}.words.srt"
    if not debug:
        common.TEMPFILENAMES.append(subtitle_original_filename)
        common.TEMPFILENAMES.append(subtitle_filtered_filename)
        common.TEMPFILENAMES.append(subtitle_filtered_forced_filename)
        common.TEMPFILENAMES.append(subtitle_filtered_previous_filename)
        common.TEMPFILENAMES.append(subtitle_words_filename)

    if current_audio2text_subtitle_version is not None and subtitle_srt_generated is not None:
        subtitle_original_idx = None
    else:
        subtitle_original_idx = (subtitle_original or {}).get(constants.K_STREAM_INDEX)
    subtitle_filtered_idx = (subtitle_filtered or {}).get(constants.K_STREAM_INDEX)
    subtitle_filtered_forced_idx = (subtitle_filtered_forced or {}).get(constants.K_STREAM_INDEX)
    subtitle_words_idx = (subtitle_words or {}).get(constants.K_STREAM_INDEX)
    audio_original_idx = audio_original.get(constants.K_STREAM_INDEX)
    audio_filtered_idx = (audio_filtered or {}).get(constants.K_STREAM_INDEX)

    logger.info(f"subtitle original = {subtitle_original_idx}, audio original = {audio_original_idx}")
    logger.info(
        f"subtitle filtered = {subtitle_filtered_idx}, subtitle filtered forced = {subtitle_filtered_forced_idx}, audio filtered = {audio_filtered_idx}, subtitle per words = {subtitle_words_idx}")

    if filter_skip:
        logger.debug("Removing filtered streams")

        # Original audio stream
        arguments.extend(["-map", f"{streams_file}:{audio_original_idx}",
                          "-c:a:0", "copy",
                          "-metadata:s:a:0", f'title={constants.TITLE_ORIGINAL}',
                          "-disposition:a:0", "default"])

        audio_output_idx = 1

        if subtitle_original_idx is not None:
            # Original subtitle stream
            arguments.extend(["-map", f"{streams_file}:{subtitle_original_idx}",
                              "-metadata:s:s:0", f'title={constants.TITLE_ORIGINAL}',
                              "-disposition:s:0", "default"])
            subtitle_output_idx = 1
        elif subtitle_srt_generated is not None:
            # Keep generated streams
            arguments.extend(["-i", subtitle_srt_generated,
                              "-map", f"{streams_file + 1}:0",
                              "-metadata:s:s:0", f'title={constants.TITLE_ORIGINAL}',
                              "-metadata:s:s:0",
                              f'{constants.K_AUDIO_TO_TEXT_VERSION}={audio_to_text_version if audio_to_text_subtitle_version else ""}',
                              "-metadata:s:s:0",
                              f'{constants.K_AUDIO_TO_TEXT_SUBTITLE_VERSION}={audio_to_text_subtitle_version if audio_to_text_subtitle_version else ""}',
                              "-metadata:s:s:0",
                              f'{constants.K_AUDIO_TO_TEXT_NOTES}={transcribe_notes if transcribe_notes else ""}',
                              "-disposition:s:0", "default"])
            subtitle_output_idx = 1
        else:
            subtitle_output_idx = 0
        if subtitle_srt_words is not None:
            arguments.extend(["-i", subtitle_srt_words,
                              "-map", f"{streams_file + 2}:0",
                              "-metadata:s:s:1", f'title={constants.TITLE_WORDS}',
                              "-metadata:s:s:1",
                              f'{constants.K_AUDIO_TO_TEXT_VERSION}={audio_to_text_version if audio_to_text_version else ""}',
                              "-disposition:s:1", "-default+metadata"])
            subtitle_output_idx += 1
        elif subtitle_words_idx is not None:
            arguments.extend(["-map", f"{streams_file}:{subtitle_words_idx}",
                              f"-metadata:s:s:{subtitle_output_idx}", f'title={constants.TITLE_WORDS}',
                              f"-disposition:s:{subtitle_output_idx}", "-default+metadata"])
            subtitle_output_idx += 1

        if common.should_replace_media_title(input_info):
            arguments.extend(
                ['-metadata', f"{constants.K_MEDIA_TITLE}={common.get_media_title_from_filename(input_info)}"])
        arguments.extend(['-metadata', f"{constants.K_MEDIA_PROCESSOR}={constants.V_MEDIA_PROCESSOR}"])
        arguments.extend(["-metadata", f"{constants.K_FILTER_HASH}="])
        arguments.extend(["-metadata", f"{constants.K_FILTER_VERSION}="])
        arguments.extend(
            ["-metadata",
             f"{constants.K_AUDIO_TO_TEXT_VERSION}={audio_to_text_version if audio_to_text_version else ''}"])
        arguments.extend(
            ["-metadata", f"{constants.K_AUDIO_TO_TEXT_FILTER}={audio_to_text_filter if audio_to_text_filter else ''}"])
        arguments.extend(
            ["-metadata",
             f"{constants.K_AUDIO_TO_TEXT_SUBTITLE_VERSION}={audio_to_text_subtitle_version if audio_to_text_subtitle_version else ''}"])
        arguments.extend(
            ["-metadata",
             f"{constants.K_AUDIO_TO_TEXT_NOTES}={transcribe_notes if transcribe_notes else ''}"])
        arguments.extend(["-metadata", f"{constants.K_FILTER_STOPPED}="])
        if mark_skip:
            arguments.extend(["-metadata", f"{constants.K_FILTER_SKIP}=true"])
        if mute_channels:
            arguments.extend(["-metadata", f"{constants.K_MUTE_CHANNELS}={mute_channels.name}"])
        arguments.extend(["-c:s", "copy"])
    else:
        subtitle_filtered_stream = 1
        arguments.extend(["-i", subtitle_filtered_filename])
        subtitle_filtered_forced_stream = 2
        arguments.extend(["-i", subtitle_filtered_forced_filename])
        input_file_index = 3

        if not debug:
            subtitle_extract_command.extend(['-v', 'quiet'])

        if subtitle_srt_generated is not None:
            subtitle_original_filename = subtitle_srt_generated
            arguments.extend(["-i", subtitle_srt_generated])
            subtitle_srt_generated_file = input_file_index
            input_file_index += 1
        else:
            subtitle_extract_command.extend(['-map', f"{streams_file}:{subtitle_original_idx}",
                                             subtitle_original_filename])
        if subtitle_filtered_idx is not None:
            subtitle_extract_command.extend(['-map', f"{streams_file}:{subtitle_filtered_idx}",
                                             subtitle_filtered_previous_filename])
        if subtitle_srt_words is not None:
            subtitle_words_filename = subtitle_srt_words
            arguments.extend(["-i", subtitle_srt_words])
            subtitle_srt_words_file = input_file_index
            input_file_index += 1
        elif subtitle_words_idx is not None:
            subtitle_extract_command.extend(['-map', f"{streams_file}:{subtitle_words_idx}",
                                             subtitle_words_filename])
        if '-map' in subtitle_extract_command:
            if verbose:
                logger.info(tools.ffmpeg.array_as_command(subtitle_extract_command))
            tools.ffmpeg.run(subtitle_extract_command, check=True)

        # subtitles
        filtered_spans = []
        stopped_spans = []
        if subtitle_codec == constants.CODEC_SUBTITLE_ASS:
            ass_data = read_ass(subtitle.clean_ssa(Path(subtitle_original_filename)))
            for event in ass_data.events:
                event.text = ASSA_TYPEFACE_REMOVE.sub('', event.text)
            if audio_to_text_subtitle_version:
                logger.info("Subtitle from transcription, subtitle alignment skipped")
            elif subtitle_words_filename and os.path.exists(subtitle_words_filename) and os.stat(
                    subtitle_words_filename).st_size > 0:
                ass_data_aligned = copy.deepcopy(ass_data)
                try:
                    aligned, aligned_stats = fix_subtitle_audio_alignment(ass_data_aligned,
                                                                          pysrt.open(subtitle_words_filename),
                                                                          lang=language,
                                                                          filename=base_filename,
                                                                          input_info=input_info)
                except ValueError as e:
                    logger.error("Cannot using aligned subtitle", e)
                    aligned = False
                    aligned_stats = None
                if debug:
                    write_ass(ass_data, Path(f"{debug_base}.aligned.{subtitle_codec}"))
                    shutil.copy(subtitle_original_filename, f"{debug_base}.original.{subtitle_codec}")
                if aligned:
                    logger.info("Using aligned subtitle: %s", aligned_stats)
                    ass_data = ass_data_aligned
                    tags[constants.K_SUB_ALIGNMENT_STATS] = aligned_stats
            else:
                logger.info("No words file, subtitle alignment skipped")
            ass_data_forced = copy.copy(ass_data)
            ass_data_forced.events = AssEventList()
            filter_progress = progress.progress(f"{base_filename} filtering", 0, len(list(ass_data.events)))
            for event_idx, event in enumerate(ass_data.events):
                original_text = event.text
                filtered_text, stopped = filter_text(censor_list, stop_list, allow_list, original_text)
                filter_progress.progress(event_idx)
                if filtered_text != original_text:
                    event.text = filtered_text
                    ass_data_forced.events.append(copy.copy(event))
                    filtered_spans.append([event.start, event.end])
                    if stopped:
                        stopped_spans.append([event.start, event.end])
            write_ass(ass_data, Path(subtitle_filtered_filename))
            write_ass(ass_data_forced, Path(subtitle_filtered_forced_filename))
            filter_progress.stop()
        elif subtitle_codec in [constants.CODEC_SUBTITLE_SRT, constants.CODEC_SUBTITLE_SUBRIP]:
            srt_data = pysrt.open(subtitle_original_filename)
            if audio_to_text_subtitle_version:
                logger.info("Subtitle from transcription, subtitle alignment skipped")
            elif subtitle_words_filename and os.path.exists(subtitle_words_filename) and os.stat(
                    subtitle_words_filename).st_size > 0:
                srt_data_aligned = copy.deepcopy(srt_data)
                try:
                    aligned, aligned_stats = fix_subtitle_audio_alignment(srt_data, pysrt.open(subtitle_words_filename),
                                                                          lang=language,
                                                                          filename=base_filename,
                                                                          input_info=input_info)
                except ValueError as e:
                    logger.error("Cannot using aligned subtitle", e)
                    aligned = False
                    aligned_stats = None
                if debug:
                    srt_data.save(Path(f"{debug_base}.aligned.{subtitle_codec}"), 'utf-8')
                    shutil.copy(subtitle_original_filename, f"{debug_base}.original.{subtitle_codec}")
                if aligned:
                    logger.info("Using aligned subtitle: %s", aligned_stats)
                    srt_data = srt_data_aligned
                    tags[constants.K_SUB_ALIGNMENT_STATS] = aligned_stats
            else:
                logger.info("No words file, subtitle alignment skipped")
            srt_data_forced = copy.copy(srt_data)
            srt_data_forced.data = []
            filter_progress = progress.progress(f"{base_filename} filtering", 0, len(list(srt_data)))
            for event_idx, event in enumerate(srt_data):
                original_text = event.text
                filtered_text, stopped = filter_text(censor_list, stop_list, allow_list, original_text)
                filter_progress.progress(event_idx)
                if filtered_text != original_text:
                    event.text = filtered_text
                    srt_data_forced.data.append(event)
                    filtered_spans.append([event.start.ordinal, event.end.ordinal])
                    if stopped:
                        stopped_spans.append([event.start.ordinal, event.end.ordinal])
            if len(srt_data.data) == 0:
                srt_data.data.append(SubRipItem())
            srt_data.save(Path(subtitle_filtered_filename), 'utf-8')
            # Forced SRT file will be invalid unless we have at least one entry
            if len(srt_data_forced.data) == 0:
                srt_data_forced.data.append(SubRipItem())
            srt_data_forced.save(Path(subtitle_filtered_forced_filename), 'utf-8')
            filter_progress.stop()
        else:
            logger.info(f"Unknown subtitle codec {subtitle_codec}")
            return CMD_RESULT_ERROR

        if len(stopped_spans) > 0:
            tags[constants.K_FILTER_STOPPED] = span_list_to_str(stopped_spans)
        if constants.K_FILTER_SKIP in tags:
            del tags[constants.K_FILTER_SKIP]
        common.write_mkv_tags(tags, tags_filename)

        if subtitle_filtered_idx is not None:
            # We are re-running the filter. Check the previous and current filtered subtitles.
            # If there is no filtering, mark as changed to remove the filtered streams
            filtered_changed = not subtitle.cmp_subtitle_text(subtitle_codec, subtitle_filtered_filename,
                                                              subtitle_filtered_previous_filename) or len(
                filtered_spans) == 0
        else:
            # Not yet filtered, see if we need to filter it
            filtered_changed = len(filtered_spans) > 0
        # If no changes, apply hash attribute
        # If we did OCR on subtitles, don't lose that, transcode to include it without the filtered streams
        if not force and not filtered_changed and not subtitle_srt_generated and not subtitle_srt_words and current_filter_version in [
            str(FILTER_VERSION),
            None, '']:
            logger.info("%s No changes, updating filter hash", base_filename)
            if not dry_run and not debug:
                tools.mkvpropedit.run([filename, "--tags", f"global:{tags_filename}"])
            return CMD_RESULT_MARKED

        arguments.extend(["-metadata", f"{constants.K_FILTER_HASH}={filter_hash}"])
        arguments.extend(["-metadata", f"{constants.K_FILTER_VERSION}={FILTER_VERSION}"])
        arguments.extend(
            ["-metadata",
             f"{constants.K_AUDIO_TO_TEXT_VERSION}={audio_to_text_version if audio_to_text_version else ''}"])
        arguments.extend(
            ["-metadata",
             f"{constants.K_AUDIO_TO_TEXT_NOTES}={transcribe_notes if transcribe_notes else ''}"])
        arguments.extend(
            ["-metadata", f"{constants.K_AUDIO_TO_TEXT_FILTER}={audio_to_text_filter if audio_to_text_filter else ''}"])
        if len(stopped_spans) > 0:
            arguments.extend(["-metadata", f"{constants.K_FILTER_STOPPED}={span_list_to_str(stopped_spans)}"])
        arguments.extend(["-metadata", f"{constants.K_FILTER_SKIP}="])
        if common.should_replace_media_title(input_info):
            arguments.extend(
                ['-metadata', f"{constants.K_MEDIA_TITLE}={common.get_media_title_from_filename(input_info)}"])
        arguments.extend(['-metadata', f"{constants.K_MEDIA_PROCESSOR}={constants.V_MEDIA_PROCESSOR}"])
        if constants.K_SUB_ALIGNMENT_STATS in tags:
            arguments.extend(
                ["-metadata", f"{constants.K_SUB_ALIGNMENT_STATS}={tags[constants.K_SUB_ALIGNMENT_STATS]}"])
        arguments.extend(["-c:s", "copy"])

        # Filtered audio stream
        audio_output_idx = 0
        # For surround sound, filter only the channel in which we expect speech
        # https://stackoverflow.com/questions/33533401/volume-adjust-and-channel-merge-on-video-using-ffmpeg
        if len(filtered_spans) > 0:
            mute_filters = []
            for span in filtered_spans:
                if isinstance(span[0], int):
                    mute_filters.append(f"volume=enable='between(t,{span[0] / 1000.0},{span[1] / 1000.0})':volume=0")
                else:
                    mute_filters.append(f"volume=enable='between(t,{span[0] / 1000.0},{span[1] / 1000.0})':volume=0")

            common.map_opus_audio_stream(arguments, input_info['streams'][audio_original_idx],
                                         audio_stream_idx=streams_file,
                                         output_stream_spec=f'a:{audio_output_idx}',
                                         audio_filters=mute_filters, mute_channels=mute_channels)
            arguments.extend([f"-metadata:s:a:{audio_output_idx}", f'title={constants.TITLE_FILTERED}',
                              f"-disposition:a:{audio_output_idx}", "default"])
            audio_output_idx += 1

        # Original audio stream
        arguments.extend(["-map", f"{streams_file}:{audio_original_idx}",
                          f"-c:a:{audio_output_idx}", "copy",
                          f"-metadata:s:a:{audio_output_idx}", f'title={constants.TITLE_ORIGINAL}'])
        if audio_output_idx == 0:
            arguments.extend([f"-disposition:a:{audio_output_idx}", "default"])
        else:
            arguments.extend([f"-disposition:a:{audio_output_idx}", "0"])
        audio_output_idx += 1

        # Filtered subtitle stream
        subtitle_output_idx = 0
        if len(filtered_spans) > 0:
            arguments.extend(["-map", f"{subtitle_filtered_stream}:0",
                              f"-metadata:s:s:{subtitle_output_idx}", f'title={constants.TITLE_FILTERED}',
                              f"-metadata:s:s:{subtitle_output_idx}", f'language={language}',
                              f"-disposition:s:{subtitle_output_idx}", "default"])
            subtitle_output_idx += 1
            arguments.extend(["-map", f"{subtitle_filtered_forced_stream}:0",
                              f"-metadata:s:s:{subtitle_output_idx}", f'title={constants.TITLE_FILTERED_FORCED}',
                              f"-metadata:s:s:{subtitle_output_idx}", f'language={language}',
                              f"-disposition:s:{subtitle_output_idx}", "-default+forced"])
            subtitle_output_idx += 1

        # Original subtitle stream
        if subtitle_srt_generated is not None:
            arguments.extend([
                "-map", f"{subtitle_srt_generated_file}:0",
                f"-metadata:s:s:{subtitle_output_idx}",
                f'{constants.K_AUDIO_TO_TEXT_VERSION}={audio_to_text_version if audio_to_text_subtitle_version else ""}',
                f"-metadata:s:s:{subtitle_output_idx}",
                f'{constants.K_AUDIO_TO_TEXT_SUBTITLE_VERSION}={audio_to_text_subtitle_version if audio_to_text_subtitle_version else ""}',
                f"-metadata:s:s:{subtitle_output_idx}",
                f'{constants.K_AUDIO_TO_TEXT_NOTES}={transcribe_notes if transcribe_notes else ""}',
            ])
        elif subtitle_original_idx is not None:
            arguments.extend(["-map", f"{streams_file}:{subtitle_original_idx}"])

        arguments.extend([f"-metadata:s:s:{subtitle_output_idx}", f'title={constants.TITLE_ORIGINAL}',
                          f"-metadata:s:s:{subtitle_output_idx}", f'language={language}'])
        if subtitle_output_idx == 0:
            arguments.extend([f"-disposition:s:{subtitle_output_idx}", "default"])
        else:
            arguments.extend([f"-disposition:s:{subtitle_output_idx}", "0"])
        subtitle_output_idx += 1

        if subtitle_srt_words is not None:
            arguments.extend(["-map", f"{subtitle_srt_words_file}:0"])
            arguments.extend([f"-metadata:s:s:{subtitle_output_idx}", f'title={constants.TITLE_WORDS}',
                              f"-metadata:s:s:{subtitle_output_idx}",
                              f'{constants.K_AUDIO_TO_TEXT_VERSION}={audio_to_text_version if audio_to_text_version else ""}',
                              f"-metadata:s:s:{subtitle_output_idx}", f'language={language}',
                              f"-disposition:s:{subtitle_output_idx}", "-default+metadata"])
            subtitle_output_idx += 1
        elif subtitle_words_idx is not None:
            arguments.extend(["-map", f"{streams_file}:{subtitle_words_idx}"])
            arguments.extend([f"-metadata:s:s:{subtitle_output_idx}", f'title={constants.TITLE_WORDS}',
                              f"-metadata:s:s:{subtitle_output_idx}",
                              f'{constants.K_AUDIO_TO_TEXT_VERSION}={audio_to_text_version if audio_to_text_version else ""}',
                              f"-metadata:s:s:{subtitle_output_idx}", f'language={language}',
                              f"-disposition:s:{subtitle_output_idx}", "-default+metadata"])
            subtitle_output_idx += 1

    # Remaining audio streams
    for extra_audio in list(filter(lambda stream: stream[constants.K_CODEC_TYPE] == constants.CODEC_AUDIO and stream[
        constants.K_STREAM_INDEX] not in [audio_original_idx, audio_filtered_idx], input_info['streams'])):
        logger.debug(f"Extra audio stream: {extra_audio}")
        arguments.extend(["-map", f"{streams_file}:{extra_audio[constants.K_STREAM_INDEX]}",
                          f"-c:a:{audio_output_idx}", "copy",
                          f"-disposition:a:{audio_output_idx}", "0"])
        audio_output_idx += 1

    # Remaining subtitle streams
    for extra_subtitle in list(
            filter(lambda stream: stream[constants.K_CODEC_TYPE] == constants.CODEC_SUBTITLE and stream[
                constants.K_STREAM_INDEX] not in [
                                      (subtitle_original or {}).get(constants.K_STREAM_INDEX),
                                      subtitle_filtered_idx,
                                      subtitle_filtered_forced_idx,
                                      subtitle_words_idx],
                                   input_info['streams'])):
        logger.debug(f"Extra subtitle stream: {extra_subtitle}")
        arguments.extend(["-map", f"{streams_file}:{extra_subtitle[constants.K_STREAM_INDEX]}",
                          f"-disposition:s:{subtitle_output_idx}", "0"])
        subtitle_output_idx += 1

    arguments.extend(["-avoid_negative_ts", "disabled", "-start_at_zero", "-copyts", "-async", "1"])
    if input_type == "mkv":
        arguments.extend(["-max_interleave_delta", "0"])

    # Video
    arguments.extend(["-c:v", "copy", "-map", f"{streams_file}:v?"])

    # Remove closed captions because they are unfiltered
    if not filter_skip:
        video_info = common.find_video_stream(input_info)
        has_closed_captions = video_info and video_info.get('closed_captions', 0) > 0
        if has_closed_captions:
            input_video_codec = common.resolve_video_codec(video_info['codec_name'])
            if input_video_codec == 'h264':
                arguments.extend(["-bsf:v", "filter_units=remove_types=6"])

    # Metadata
    arguments.extend(
        ["-map_chapters", str(streams_file), "-map_metadata", str(streams_file),
         "-c:t", "copy", "-map", f"{streams_file}:t?"])

    # Fixes: Too many packets buffered for output stream (can be due to late start subtitle)
    # http://stackoverflow.com/questions/49686244/ddg#50262835
    arguments.extend(['-max_muxing_queue_size', '1024'])

    arguments.append(temp_filename)

    if dry_run:
        logger.info(tools.ffmpeg.array_as_command(arguments))
        return CMD_RESULT_FILTERED

    # Check again because there is time between the initial check and when we write to the file
    if common.assert_not_transcoding(input_file, exit=False) != 0:
        return CMD_RESULT_ERROR
    try:
        Path(temp_filename).touch(mode=0o664, exist_ok=False)
    except FileExistsError:
        return CMD_RESULT_ERROR

    logger.info(f"Starting filtering of {filename} to {temp_filename}")
    logger.info(tools.ffmpeg.array_as_command(arguments))
    tools.ffmpeg.run(arguments, check=True)

    if os.stat(temp_filename).st_size == 0:
        logger.fatal(f"Output at {temp_filename} is zero length")
        return CMD_RESULT_ERROR

    #
    # Encode Done. Performing Cleanup
    #
    logger.info(f"Finished filtering of {filename} to {temp_filename}")

    fsutil.match_owner_and_perm(target_path=temp_filename, source_path=filename)

    # Hide original file in case OUTPUT_TYPE is the same as input
    if not debug:
        os.replace(filename, hidden_filename)
    try:
        os.replace(temp_filename, output_filename)
    except OSError:
        # Put original file back as fall back
        os.replace(hidden_filename, filename)
        logger.fatal(f"Failed to move converted file: {temp_filename}")
        return CMD_RESULT_ERROR

    if not keep and not debug:
        os.remove(hidden_filename)

    logger.info("Filtering done")

    return CMD_RESULT_FILTERED


def load_censor_list() -> list[re.Pattern]:
    # TODO: allow multiple lists based on 'levels' (R, PG, G ?) and allow selection at runtime
    result = loadtxt(os.path.join(os.path.dirname(common.__file__), 'censor_list.txt'), dtype='str', delimiter='\xFF')
    result = list(filter(phrase_list_accept_condition, result))
    sort_sub = re.compile('[^A-Za-z]+')
    result.sort(key=lambda e: len(sort_sub.sub('', e)), reverse=True)
    result = list(map(lambda e: phrase_to_pattern(e), result))
    result = list(map(lambda e: re.compile(e, flags=re.IGNORECASE), result))
    return result


def load_stop_list() -> list[re.Pattern]:
    # TODO: allow multiple lists based on 'levels' (R, PG, G ?) and allow selection at runtime
    result = loadtxt(os.path.join(os.path.dirname(common.__file__), 'stop_list.txt'), dtype='str', delimiter='\xFF')
    result = list(filter(phrase_list_accept_condition, result))
    sort_sub = re.compile('[^A-Za-z]+')
    result.sort(key=lambda e: len(sort_sub.sub('', e)), reverse=True)
    result = list(map(lambda e: phrase_to_pattern(e), result))
    result = list(map(lambda e: re.compile(e, flags=re.IGNORECASE), result))
    return result


def load_allow_list() -> list[re.Pattern]:
    result = loadtxt(os.path.join(os.path.dirname(common.__file__), 'allow_list.txt'), dtype='str', delimiter='\xFF')
    result = list(filter(phrase_list_accept_condition, result))
    sort_sub = re.compile('[^A-Za-z]+')
    result.sort(key=lambda e: len(sort_sub.sub('', e)), reverse=True)
    result = list(map(lambda e: phrase_to_pattern(e), result))
    result = list(map(lambda e: re.compile(e, flags=re.IGNORECASE), result))
    return result


def compute_filter_hash(censor_list=None, stop_list=None, allow_list=None):
    if censor_list is None:
        censor_list = load_censor_list()
    if stop_list is None:
        stop_list = load_stop_list()
    if allow_list is None:
        allow_list = load_allow_list()
    hash_input = str(FILTER_VERSION) \
                 + ','.join(list(map(lambda r: r.pattern, censor_list))) \
                 + ','.join(list(map(lambda r: r.pattern, stop_list))) \
                 + ','.join(list(map(lambda r: r.pattern, allow_list)))
    filter_hash = hashlib.sha512(hash_input.encode("utf-8")).hexdigest()
    logger.info(f"expected filter hash = {filter_hash}, expected filter version = {FILTER_VERSION}")
    return filter_hash


def is_filter_version_outdated(tags: dict[str, str]) -> bool:
    """
    Determines if the filter version is out of date, indicating more work than usual to update the filter.
    :param tags:
    :return: True if the version is out of date or missing.
    """
    if tags is None:
        return True
    current_filter_version = int(tags.get(constants.K_FILTER_VERSION, "0"))
    if current_filter_version < FILTER_VERSION:
        return True
    if tags.get(constants.K_AUDIO_TO_TEXT_VERSION, None):
        current_audio2text_version = int(tags.get(constants.K_AUDIO_TO_TEXT_VERSION))
        if current_audio2text_version < AUDIO_TO_TEXT_VERSION:
            return True
    if tags.get(constants.K_AUDIO_TO_TEXT_SUBTITLE_VERSION, None):
        current_audio2text_subtitle_version = int(tags.get(constants.K_AUDIO_TO_TEXT_SUBTITLE_VERSION))
        if current_audio2text_subtitle_version < AUDIO_TO_TEXT_SUBTITLE_VERSION:
            return True
    return False


def need_original_subtitle_ocr(subtitle_original: dict, media_duration: float, force: bool) -> bool:
    """
    Determine if the original subtitle needs ocr from image subtitles.
    :param subtitle_original:
    :return: True to perform OCR
    """
    if not subtitle_original:
        return True

    duration_s = subtitle_original.get('tags', {}).get('DURATION', '')
    try:
        if duration_s:
            duration = edl_util.parse_edl_ts(duration_s)
            if duration < (media_duration * 0.60):
                return True
    except Exception:
        pass

    return force


def need_words_transcribed(subtitle_words: dict, current_audio2text_version: str, force: bool) -> bool:
    """
    Determine if the original subtitle needs transcribed from audio.
    :param current_audio2text_version:
    :return: True for words needed
    """
    if force:
        return True
    if not subtitle_words:
        return True
    if not current_audio2text_version:
        return True
    return int(current_audio2text_version) < AUDIO_TO_TEXT_VERSION


def need_original_subtitle_transcribed(subtitle_original: dict, current_audio2text_subtitle_version: str,
                                       media_duration: float, force: bool) -> bool:
    """
    Determine if the original subtitle needs transcribed from audio.
    :param subtitle_original:
    :return: True for original needed, True for words needed
    """

    if not subtitle_original:
        return True

    try:
        duration_s = subtitle_original.get('tags', {}).get('DURATION', '')
        if duration_s:
            duration = edl_util.parse_edl_ts(duration_s)
            if duration < (media_duration * 0.60):
                return True
    except Exception:
        pass

    if not current_audio2text_subtitle_version:
        # current subtitle wasn't transcribed
        return False

    if force:
        return True

    return int(current_audio2text_subtitle_version) < AUDIO_TO_TEXT_SUBTITLE_VERSION


def _open_subtitle_stream(input_info: dict, subtitle_info: dict) -> subprocess.Popen[str]:
    args = ['-nostdin', "-loglevel", "error",
            '-i', input_info['format']['filename'],
            '-map', f'0:{subtitle_info[constants.K_STREAM_INDEX]}',
            '-c', 'copy',
            '-f', 'srt', '-']
    logger.info('ffmpeg %s', ' '.join(args))
    return tools.ffmpeg.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1, text=True)


PATTERN_DETECT_TRANSCRIBED_BY_VERSION_3 = re.compile(r'[A-Z.,]')


def detect_transcribed_by_version_3(current_audio2text_version: str, input_info: dict, subtitle_original: dict):
    """
    In audio version 3 we didn't always set the stream audio to text version. Imply it here.
    :param current_audio2text_version:
    :param input_info:
    :param subtitle_original:
    :return: True of subtitle was transcribed by version 3
    """
    if current_audio2text_version == "3" and subtitle_original and subtitle_original.get(constants.K_TAGS, {}).get(
            constants.K_AUDIO_TO_TEXT_VERSION, '') == '' \
            and subtitle_original.get(constants.K_CODEC_NAME, '') == constants.CODEC_SUBTITLE_SUBRIP \
            and constants.K_STREAM_INDEX in subtitle_original:
        logger.info("Checking original subtitle to determine if it was transcribed in version 3")
        # extract and check value for no caps and no punctuation
        subtitle_original_proc = _open_subtitle_stream(input_info, subtitle_original)
        transcribed = True
        for sub in pysrt.stream(subtitle_original_proc.stdout):
            if PATTERN_DETECT_TRANSCRIBED_BY_VERSION_3.search(sub.text):
                transcribed = False
                break
        if transcribed:
            logger.info("Detected original subtitle was transcribed in version 3")
        subtitle_original_proc.stdout.close()
        if subtitle_original_proc.wait() in [130, 255]:
            raise KeyboardInterrupt()
        return transcribed
    return False


def ocr_subtitle_bitmap_to_srt(input_info, temp_base, language=None, verbose=False):
    """
    Attempts to find a bitmap subtitle stream and run OCR to generate an SRT file.
    :return: file name as string or None
    """
    global debug

    subtitle_filename = None

    subtitle_srt_filename = f"{temp_base}.srt"
    if not debug:
        common.TEMPFILENAMES.append(subtitle_srt_filename)
    if os.access(subtitle_srt_filename, os.R_OK):
        if debug:
            return subtitle_srt_filename
        else:
            os.remove(subtitle_srt_filename)

    bluray = find_subtitle_bluray(input_info, language)
    if bluray:
        bluray = bluray[0]
        subtitle_filename = f"{temp_base}.sup"
        if not debug:
            common.TEMPFILENAMES.append(subtitle_filename)
        extract_command = ['-hide_banner', '-y', '-analyzeduration',
                           common.ANALYZE_DURATION,
                           '-probesize', common.PROBE_SIZE,
                           '-i', input_info['format']['filename'],
                           '-map', f'0:{bluray[constants.K_STREAM_INDEX]}', '-c:s', 'copy',
                           subtitle_filename
                           ]
        if verbose:
            logger.info(tools.ffmpeg.array_as_command(extract_command))
        tools.ffmpeg.run(extract_command, check=True)

    dvd = find_subtitle_dvdsub(input_info, language)
    if dvd:
        dvd = dvd[0]
        subtitle_filename = f"{temp_base}.vob"
        if not debug:
            common.TEMPFILENAMES.append(subtitle_filename)
        extract_command = ['-hide_banner', '-y', '-analyzeduration',
                           common.ANALYZE_DURATION,
                           '-probesize', common.PROBE_SIZE,
                           '-i', input_info['format']['filename'],
                           '-map', f'0:{dvd[constants.K_STREAM_INDEX]}', '-c:s', 'dvdsub',
                           subtitle_filename
                           ]
        if verbose:
            logger.info(tools.ffmpeg.array_as_command(extract_command))
        tools.ffmpeg.run(extract_command, check=True)
        # Add an idx file to remove transparency. OCR doesn't like transparency.
        subtitle_idx_filename = f"{temp_base}.idx"
        if not debug:
            common.TEMPFILENAMES.append(subtitle_idx_filename)
        width, height = common.get_video_width_height(input_info)
        create_subtitle_idx_file(subtitle_idx_filename, width, height)

    if subtitle_filename is None:
        return None

    tools.subtitle_edit.run([subtitle_filename], check=True)
    if not os.access(subtitle_srt_filename, os.R_OK):
        logger.error("SRT not generated from OCR")
        return None

    word_found_pct = words_in_dictionary_pct(subtitle_srt_filename, language,
                                             float(input_info[constants.K_FORMAT][constants.K_DURATION]))
    if word_found_pct < WORD_FOUND_PCT_THRESHOLD:
        logger.error(f"OCR text appears to be incorrect, {word_found_pct}% words found in {language} dictionary")
        return None

    return subtitle_srt_filename


def whisper_language(language: str) -> str:
    """
    Get the language code for the Whisper transcriber from the three character language code.
    :param language: three character language code
    :returns: language code appropriate for Whisper
    """
    if language == 'spa':
        return 'es'
    if language in ['eng', 'en']:
        return 'en'
    return language[0:2]


def audio_to_words_srt(input_info: dict, audio_original: dict, workdir, audio_filter: str = None, language=None) \
        -> (Union[str, None], Union[str, None]):
    """
    Create a SRT subtitle with an event for each word.
    :return: srt filename for words
    """
    global debug

    freq = 48000
    num_channels = 1  # num_channels 2 doesn't work (2024-02-17), it takes the input as mono

    temp_fd, words_filename = tempfile.mkstemp(dir=workdir, suffix='.srt')
    os.close(temp_fd)
    if not debug:
        common.TEMPFILENAMES.append(words_filename)

    temp_fd, audio_filename = tempfile.mkstemp(dir=workdir, suffix='.wav')
    os.close(temp_fd)
    if not debug:
        common.TEMPFILENAMES.append(audio_filename)

    extract_command = ['-y', '-nostdin', "-loglevel", "error",
                       '-i', input_info['format']['filename'],
                       '-map', f'0:{audio_original[constants.K_STREAM_INDEX]}',
                       '-ar', str(freq)]
    if audio_filter:
        if 'pan=' in audio_filter:
            if 'stereo' in audio_filter:
                num_channels = 2
            else:
                num_channels = 1
        else:
            extract_command.extend(['-ac', str(num_channels)])
        extract_command.extend(['-af', audio_filter])
    else:
        extract_command.extend(['-ac', str(num_channels)])
    extract_command.extend(['-f', 'wav', audio_filename])

    whisper_result = []
    audio_progress = StreamCapturingProgress(
        "stderr",
        progress.progress(
            f"{os.path.basename(input_info['format']['filename'])} transcription",
            0, 100))
    logger.debug(tools.ffmpeg.array_as_command(extract_command))
    try:
        tools.ffmpeg.run(extract_command, check=True)

        for whisper_model_idx, whisper_model in enumerate(lazy_get_whisper_models()):
            whisper_result = whisper_model.transcribe(
                audio=audio_filename,
                language=whisper_language(language),
                fp16=False,
                word_timestamps=True,
                verbose=False,
                beam_size=WHISPER_BEAM_SIZE,
                patience=WHISPER_PATIENCE,
            )

            # need to check for words, sometimes only music symbols are returned
            unique_words = set()
            for segment in whisper_result.get("segments", []):
                for word in segment.get("words", []):
                    word_text = ''.join(filter(lambda e: e.isalpha(), word['word']))
                    if len(word_text) > 0:
                        unique_words.add(word_text)
            if len(unique_words) > 100:
                break
            elif whisper_model_idx == 0:
                logger.warning(f"Transcribing produced {len(unique_words)} words, retrying with a different model")
    finally:
        audio_progress.finish()
        if not debug:
            os.remove(audio_filename)

    confs = []
    subs_words = []
    for segment in whisper_result["segments"]:
        for word in segment["words"]:
            if 'probability' in word:
                confs.append(word['probability'])
            s = SubRipItem(index=len(subs_words),
                           start=SubRipTime(seconds=word['start']),
                           end=SubRipTime(seconds=word['end']),
                           text=word['word'].strip())
            subs_words.append(s)

    # allow for no words, some videos don't have speech
    if len(subs_words) == 0:
        logger.warning("audio-to-text transcription empty")
        # return None

    srt_words = SubRipFile(items=subs_words, path=words_filename)
    srt_words.save(Path(words_filename), 'utf-8')

    if len(confs) > 5:
        conf_min = min(confs)
        conf_max = max(confs)
        conf_avg = mean(confs)
        conf_stdev = stdev(confs)
        conf_notes = f'freq {freq} af {audio_filter} conf [{conf_min:.3f},{conf_max:.3f}] {conf_avg:.3f}σ{conf_stdev:.3f}'
    else:
        conf_notes = ""

    conf_notes = f"{conf_notes} whisper(model={WHISPER_MODEL_NAME},beam_size={WHISPER_BEAM_SIZE},patience={WHISPER_PATIENCE})"

    return words_filename, conf_notes


def words_to_subtitle_srt(input_info: dict, words_filename: str, workdir, language=None) -> Union[str, None]:
    """
    Create a SRT subtitle from transcribed words.
    :return: srt filename
    """
    temp_fd, subtitle_srt_filename = tempfile.mkstemp(dir=workdir, suffix='.srt')
    os.close(temp_fd)
    if not debug:
        common.TEMPFILENAMES.append(subtitle_srt_filename)

    subs_words = pysrt.open(Path(words_filename), 'utf-8')
    subs_text = srt_words_to_sentences(subs_words, language)
    srt = SubRipFile(items=subs_text, path=subtitle_srt_filename)
    srt.save(Path(subtitle_srt_filename), 'utf-8')

    word_found_pct = words_in_dictionary_pct(subtitle_srt_filename, language,
                                             float(input_info[constants.K_FORMAT][constants.K_DURATION]))
    if word_found_pct < WORD_FOUND_PCT_THRESHOLD:
        logger.error(
            f"audio-to-text transcription appears to be incorrect, {word_found_pct}% words found in {language} dictionary")
        return None

    return subtitle_srt_filename


class LangToolSpellchecker(object):
    def __init__(self, lang_tool: language_tool_python.LanguageTool):
        self._lang_tool = lang_tool

    def spell(self, word: str) -> bool:
        for match in self._lang_tool.check(word):
            if match.ruleIssueType == 'misspelling' and match.ruleId.startswith('MORFOLOGIK_RULE'):
                return False
        return True


def get_spell_checker(language: str):
    return LangToolSpellchecker(_get_lang_tool(language))


PATTERN_WORDS_IN_DICT_SPLIT = re.compile('[^A-Za-z\' ]+')


def words_in_dictionary_pct(subtitle_srt_filename, language: str, duration: float) -> float:
    spellchecker = get_spell_checker(language)
    if spellchecker is None:
        logger.warning("spell checker not found, skipping dictionary check")
        return 100.0

    word_count = 0
    word_found_count = 0
    srt_data = pysrt.open(subtitle_srt_filename)
    for event in list(srt_data):
        for word in PATTERN_WORDS_IN_DICT_SPLIT.sub(' ', event.text).split():
            word_count += 1
            if spellchecker.spell(word):
                word_found_count += 1
    if word_count < 100 and duration > 630:
        logger.warning(f"word count less than 100 for duration {duration}, returning 0%")
        return 0.0
    word_found_pct = ceil(100.0 * float(word_found_count) / (float(word_count) + 0.001))
    logger.info(f"SRT words = {word_count}, found = {word_found_count}, {word_found_pct}%")
    return word_found_pct


def find_subtitle_dvdsub(input_info, language=None):
    return one_or_zero_streams(
        common.find_streams_by_codec_and_language(input_info, constants.CODEC_SUBTITLE,
                                                  [constants.CODEC_SUBTITLE_DVDSUB],
                                                  language))


def find_subtitle_bluray(input_info, language=None):
    return one_or_zero_streams(
        common.find_streams_by_codec_and_language(input_info, constants.CODEC_SUBTITLE,
                                                  [constants.CODEC_SUBTITLE_BLURAY],
                                                  language))


def one_or_zero_streams(streams):
    default_streams = list(
        filter(lambda stream: stream.get('disposition') and stream.get('disposition').get('default') > 0, streams))
    if len(default_streams) > 0:
        return default_streams[0:1]
    return streams[0:1]


def span_list_to_str(span_list: list) -> str:
    if span_list is None:
        return ''
    return ','.join(
        map(lambda span: common.s_to_ts(span[0] / 1000.0) + '-' + common.s_to_ts(span[1] / 1000.0), span_list))


STOP_CLEAN_PATTERN = re.compile("^((?:[{].*?[}])*)")
# Identifies a word break in both SRT and ASS. ASS is more complex than SRT and has it's own markup.
WORD_BREAKS = r'((?:\\s|[,!]|[{].+?[}]|[\\\\]x[0-9a-fA-F]+|[\\\\][Nh])+|\\b|$)'
# When the censor list uses the regex "^" for the beginning of text, this is substituted to make it work
NO_PREVIOUS_WORD = r'(?<!\\w\\s)'
# Matches subtitles that have text already filtered using various symbols
PRE_FILTERED = re.compile(r'[!@#$%^&*+]\s?(?:[@#$%^&*+]\s?){2,}', flags=re.IGNORECASE)
# The string to use for masking
MASK_STR = '***'
TRAILING_PUNCUATION = re.compile(r'\*\*\*([,!.?\'’]+)')
REDUNANT_REPLACEMENTS = re.compile(r'(\*\*\*[\s,]*)+\*\*\*')


def contains_pattern_repl(matchobj, allow_ranges: list[tuple]) -> str:
    original = matchobj.group(0)
    for allow_range in allow_ranges:
        if allow_range[0] <= matchobj.start(0) < allow_range[1] \
                or allow_range[0] <= matchobj.end(0) < allow_range[1]:
            return original

    masked = ''
    logger.debug("%s", str(matchobj.groups()))
    for group_idx in range(1, len(matchobj.groups()) + 1):
        if group_idx % 2 == 0:
            masked = masked + MASK_STR
        elif matchobj.group(group_idx) is not None:
            masked = masked + matchobj.group(group_idx)
        pass
    return masked


def matches_stop_pattern(stop_pattern: re.Pattern, text: str, allow_ranges: list[tuple]) -> bool:
    for m in stop_pattern.finditer(text):
        allowed = False
        for allow_range in allow_ranges:
            if allow_range[0] <= m.start(0) < allow_range[1] \
                    or allow_range[0] <= m.end(0) < allow_range[1]:
                allowed = True
                break
        if not allowed:
            return True
    return False


def filter_text(censor_list: list[re.Pattern], stop_list: list[re.Pattern], allow_list: list[re.Pattern], text) -> \
        Tuple[str, bool]:
    """
    Filter text using the lists.
    :param censor_list: phrases in the censor list are removed from the text, such as adjectives or exclamations
    :param stop_list: phrases in the stop list indicate suspicious subject and all text is removed
    :param allow_list: phrases to allow that match censor_list
    :param text:
    :return: filtered text, True if phrase in stop list was found
    """

    allow_ranges: list[tuple] = []
    for allow_pattern in allow_list:
        try:
            for m in allow_pattern.finditer(text):
                range, group = get_allow_range(m)
                allow_ranges.append(range)
        except re.error as e:
            print(f"ERROR in allow list: {allow_pattern.pattern}")
            raise e

    if PRE_FILTERED.search(text):
        text2 = PRE_FILTERED.sub(MASK_STR, text)
        if text2 == text:
            # We need the text to change to mute the audio
            text = text2 + ' _'
        else:
            text = text2
    for stop_pattern in stop_list:
        try:
            if matches_stop_pattern(stop_pattern, text, allow_ranges):
                text = STOP_CLEAN_PATTERN.search(text).expand(fr"\1{MASK_STR}")
                return text, True
        except re.error as e:
            print(f"ERROR in stop list: {stop_pattern.pattern}")
            raise e
    for contains_pattern in censor_list:
        try:
            text = contains_pattern.sub(lambda m: contains_pattern_repl(m, allow_ranges), text)
        except re.error as e:
            print(f"ERROR in censor list: {contains_pattern.pattern}")
            raise e
    # clean up trailing punctuation
    text = TRAILING_PUNCUATION.sub(MASK_STR, text)
    # clean up redundant replacements
    text = REDUNANT_REPLACEMENTS.sub(MASK_STR, text)
    return text, False


def get_allow_range(m: re.Match[str]) -> tuple[tuple[int, int], str]:
    original = m.string
    # print(f"get_allow_range(): original = {original}")
    begin = m.span(0)[0]
    end = m.span(0)[1]
    # print(f"get_allow_range(): begin = {begin}, end = {end}")
    while begin < (end - 1) and original[begin].isspace():
        begin += 1
    while begin < (end - 1) and original[end - 1].isspace():
        end -= 1
    # print(f"get_allow_range(): begin = {begin}, end = {end}")
    return (begin, end), original[begin:end]


PHRASE_FANCY_WORD_BREAKS = re.compile(r'([^\s^$]+)\s*', flags=re.IGNORECASE)
PHRASE_BEGINNING_MARKER = re.compile(r'[\^]')
PHRASE_STARTING_WORD_BREAK = re.compile(r'.')
PHRASE_GROUPING = re.compile(r'\(')


def phrase_to_pattern(phrase):
    """
    Transforms a phrase, which is roughly a regex, into something that will match both SRT and ASS markup.
    """
    phrase_grouping = PHRASE_GROUPING.sub('(?:', phrase)
    phrase_fancy_word_breaks = PHRASE_FANCY_WORD_BREAKS.sub(r'(\1)' + WORD_BREAKS, phrase_grouping)
    phrase_beginning_marker = PHRASE_BEGINNING_MARKER.sub(NO_PREVIOUS_WORD, phrase_fancy_word_breaks)
    phrase_starting_word_break = PHRASE_STARTING_WORD_BREAK.sub(WORD_BREAKS, '.') + phrase_beginning_marker
    return phrase_starting_word_break


def phrase_list_accept_condition(e):
    e = e.strip()
    if len(e) == 0:
        return False
    return e[0] != '#'


def _tag_as_skipped(filename: str, tags_filename: str, input_info: dict, dry_run: bool, debug: bool,
                    verbose: bool) -> int:
    """
    Mark a file to skip the filter.
    :param filename:
    :param tags_filename:
    :param input_info:
    :return: CMD_RESULT_UNCHANGED or CMD_RESULT_MARKED
    """
    tags = input_info.get(constants.K_FORMAT, {}).get(constants.K_TAGS, {}).copy()
    if common.is_truthy(tags.get(constants.K_FILTER_SKIP)):
        # already marked
        return CMD_RESULT_UNCHANGED

    tags[constants.K_FILTER_SKIP] = 'true'
    for key in [constants.K_FILTER_HASH, constants.K_FILTER_VERSION, constants.K_FILTER_STOPPED,
                constants.K_MUTE_CHANNELS]:
        if key in tags:
            del tags[key]
    if common.should_replace_media_title(input_info):
        tags[constants.K_MEDIA_TITLE] = common.get_media_title_from_filename(input_info)
    tags[constants.K_MEDIA_PROCESSOR] = constants.V_MEDIA_PROCESSOR
    common.write_mkv_tags(tags, tags_filename)
    if not dry_run and not debug:
        tools.mkvpropedit.run([filename, "--tags", f"global:{tags_filename}"])
    return CMD_RESULT_MARKED


def find_subtitle_element_idx_ge(time_ordinals: list[int], start: int) -> int:
    """
    Find leftmost element greater than or equal to x
    https://docs.python.org/3/library/bisect.html
    """
    i = bisect_left(time_ordinals, start)
    if i != len(time_ordinals):
        return i
    raise ValueError


def find_subtitle_element_idx_le(time_ordinals: list[int], start: int) -> int:
    """
    Find rightmost element less than or equal to x
    https://docs.python.org/3/library/bisect.html
    """
    i = bisect_right(time_ordinals, start)
    if i:
        return i - 1
    raise ValueError


SUBTITLE_TEXT_TO_PLAIN_REMOVE = re.compile(r"\[.*?]|\(\D.*?\)|\{.*?}|<.*?>")
SUBTITLE_TEXT_TO_PLAIN_WS = re.compile(r"\\[A-Za-z]|[,.?$!*&()\"-]|\W'|'\W|[\u007F-\uFFFF]")
SUBTITLE_TEXT_TO_PLAIN_SQUEEZE_WS = re.compile(r"\s+")
SUBTITLE_TEXT_TO_PLAIN_NUMBERS = re.compile(r"(\d+(?:\.\d+)?)")
SUBTITLE_TEXT_TO_PLAIN_ORDINALS = re.compile(r"(\d+)(?:st|nd|rd|th)")
SUBTITLE_TEXT_TO_PLAIN_CURRENCY = re.compile(r"\$([0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]+)?)")
SUBTITLE_TEXT_TO_PLAIN_ABBREV_DICT = {
    'dr.': 'doctor',
    'mr.': 'mister',
    'mrs.': 'misses',
}
SUBTITLE_TEXT_TO_PLAIN_ABBREV_PATTERN = re.compile(
    "(" + "|".join(map(lambda e: e.replace('.', '[.]'), SUBTITLE_TEXT_TO_PLAIN_ABBREV_DICT.keys())) + ")",
    flags=re.IGNORECASE)


class TextToTranscribed(object):

    def __init__(self, pattern: re.Pattern, lang: str):
        self.pattern = pattern
        self.lang = lang

    def replacements(self, match: re.Match[str]) -> list[str]:
        return [match[0]]


class FancyQuotesTranscribed(TextToTranscribed):
    fancy_pattern = re.compile(r"[\u0060\u00B4\u2018\u2019\u201C\u201D]")

    def __init__(self, lang: str):
        super().__init__(self.fancy_pattern, lang)

    def replacements(self, match: re.Match[str]) -> list[str]:
        quote = match.group(0)
        if quote in "\u0060\u00B4\u2018\u2019":
            return ["'"]
        if quote in "\u201C\u201D":
            return ['"']
        return [quote]


class ConstantReplacementTranscribed(TextToTranscribed):

    def __init__(self, pattern: re.Pattern, lang: str, replacement: str):
        super().__init__(pattern, lang)
        self.replacement = [replacement]

    def replacements(self, match: re.Match[str]) -> list[str]:
        return self.replacement


class Num2WordsTranscribed(TextToTranscribed):

    def __init__(self, pattern: re.Pattern, lang: str, to: str):
        super().__init__(pattern, lang)
        self.to = to

    def replacements(self, match: re.Match[str]) -> list[str]:
        return [num2words(match.group(1), to=self.to, lang=self.lang)]


class CardinalNumTranscribed(TextToTranscribed):

    def __init__(self, lang: str):
        super().__init__(SUBTITLE_TEXT_TO_PLAIN_NUMBERS, lang)

    def replacements(self, match: re.Match[str]) -> list[str]:
        numbers = match.group(1).replace(',', '')
        results = [num2words(numbers, lang=self.lang)]
        if numbers.isdigit():
            if len(numbers) > 2:
                # use a word per digit, such as a spoken account number
                digits = [num2words(ch, lang=self.lang) for ch in numbers]
                results.append(' '.join(digits))
            if len(numbers) == 3:
                # Match numbers like "622" = "six twenty two"
                results.append(num2words(numbers[0], lang=self.lang) + " " + num2words(numbers[1:], lang=self.lang))
            if len(numbers) == 4:
                # Match numbers like "1822" = "eighteen twenty two"
                # results.append(num2words(numbers[0:1], lang=self.lang))
                # results.append(num2words(numbers[2:], lang=self.lang))
                results.append(num2words(numbers, to='year', lang=self.lang))

        return results


class CurrencyTranscribed(TextToTranscribed):

    def __init__(self, lang: str):
        super().__init__(SUBTITLE_TEXT_TO_PLAIN_CURRENCY, lang)

    def replacements(self, match: re.Match[str]) -> list[str]:
        result = num2words(match.group(1).replace(',', ''), to='currency', lang=self.lang)
        if '$' in match.group(0) and (self.lang in ['en', 'eng'] or self.lang.endswith('_US')):
            result = result.replace('euro', 'dollar')
        return [result]


class AbbreviationsTranscribed(TextToTranscribed):
    def __init__(self, lang: str):
        super().__init__(SUBTITLE_TEXT_TO_PLAIN_ABBREV_PATTERN, lang)

    def replacements(self, match: re.Match[str]) -> list[str]:
        abbrev = match.group(1).lower()
        return [abbrev, SUBTITLE_TEXT_TO_PLAIN_ABBREV_DICT[abbrev]]


class HomonymsTranscribed(TextToTranscribed):

    def __init__(self, lang: str):
        self._homonyms: dict[str, set[str]] = {}
        self.load_homonyms(lang)
        pattern = r"|".join(map(lambda e: r"\b" + e + r"\b", self._homonyms.keys()))
        super().__init__(re.compile(pattern), lang)

    def load_homonyms(self, lang: str):
        homonyms = loadtxt(os.path.join(os.path.dirname(common.__file__), f'../resources/homonyms-{lang}.txt'),
                           dtype='str',
                           delimiter='\xFF', ndmin=1)
        slang = loadtxt(os.path.join(os.path.dirname(common.__file__), f'../resources/slang-{lang}.txt'), dtype='str',
                        delimiter='\xFF', ndmin=1)
        lines = concatenate((homonyms, slang))
        for line in filter(phrase_list_accept_condition, lines):
            words = set(filter(lambda e: len(e) > 0, map(lambda e: e.strip(), line.split(','))))
            if len(words) < 2:
                continue
            for word in words:
                if word in self._homonyms:
                    self._homonyms[word] = self._homonyms[word].union(words)
                else:
                    self._homonyms[word] = words

    def replacements(self, match: re.Match[str]) -> list[str]:
        word = match.group(0).lower()
        if word in self._homonyms:
            logger.debug("Found homonym %s : %s", word, self._homonyms[word])
            return list(self._homonyms[word])
        else:
            return [word]


TEXT_TO_TRANSCRIBED = [
    FancyQuotesTranscribed('en'),
    ConstantReplacementTranscribed(SUBTITLE_TEXT_TO_PLAIN_REMOVE, 'en', ""),
    HomonymsTranscribed('en'),
    CurrencyTranscribed('en'),
    Num2WordsTranscribed(SUBTITLE_TEXT_TO_PLAIN_ORDINALS, 'en', 'ordinal'),
    CardinalNumTranscribed('en'),
    AbbreviationsTranscribed('en'),
    ConstantReplacementTranscribed(SUBTITLE_TEXT_TO_PLAIN_WS, 'en', " "),
    ConstantReplacementTranscribed(SUBTITLE_TEXT_TO_PLAIN_SQUEEZE_WS, 'en', " "),
]


def _subtitle_text_to_plain(text: str, lang='en') -> list[str]:
    """
    Clean up subtitle text to remove non-spoken markers
    remove [ Name ]
    remove ( Name )
    remove { ASS tags }
    replace "\\N" with " "
    remove '.' from abbreviations
    all lowercase to match closer to transcription
    squeeze whitespace to match closer to transcription
    numbers to text
    """
    results = [text]

    for transform in TEXT_TO_TRANSCRIBED:
        processing: list[tuple[str, int]] = [(result, 0) for result in results]
        processed = []
        while processing:
            current = processing.pop(0)
            match = transform.pattern.search(current[0], current[1])
            if match is None:
                processed.append(current[0])
            else:
                replacements = transform.replacements(match)
                if len(replacements) > 0:
                    for replacement in replacements:
                        replaced = current[0][0:match.start(0)] + replacement + current[0][match.end(0):]
                        processing.append((replaced, match.start(0) + len(replacement)))
                else:
                    processing.append((current[0], match.end(0)))
        results = processed

    results = list(set([result.lower().strip() for result in results]))
    results.sort()
    return results


def _is_transcribed_word_suspicious(event: SubRipItem, lang: str = 'eng') -> bool:
    return _is_transcribed_word_ambiguous(event.text, lang) and event.duration.ordinal > 700


def _is_transcribed_word_ambiguous(word: str, lang: str = 'eng') -> bool:
    return word in ['the', 'in', 'is', 'an', 'yeah', 'that', 'hm', 'hmm', 'to']


ENG_VOWELS = ['a', 'e', 'i', 'o', 'u', 'y']


def _capitalize(text: str, language: str, spellchecker) -> str:
    if text in ['i', 'dr', 'dr.', 'mr', 'mr.', 'mrs', 'mrs.', 'ms', 'ms.']:
        return text.capitalize()
    if text.startswith("i'"):
        return text.capitalize()

    # single letters, probably abbreviation
    if len(text) == 1 and text not in ['a']:
        return text.capitalize()
    # multiple periods, probably abbreviation
    if text.count('.') > 1:
        return text.upper()

    if spellchecker is None:
        return text
    if language is None:
        return text
    if spellchecker.spell(text):
        return text
    # short with adjacent constants, probably abbreviation
    if 2 <= len(text) <= 3 and text[0] not in ENG_VOWELS and text[1] not in ENG_VOWELS:
        return text.upper()
    return text.capitalize()


def _is_question(sentence: str, lang: str) -> bool:
    if not sentence:
        return False
    words = sentence.split()
    if len(words) == 0:
        return False
    return words[0].lower() in ['who', 'what', 'when', 'where', 'why', 'how']


def _add_punctuation(sentence: str, lang: str) -> str:
    if not sentence:
        return sentence
    if _is_question(sentence, lang):
        return sentence + '?'
    else:
        return sentence + '.'


_LANG_TOOLS = {}
LTP_DOWNLOAD_VERSION = '6.6'


def _get_lang_tool(language: str) -> Union[None, language_tool_python.LanguageTool]:
    global _LANG_TOOLS
    lang_tool_lang = 'en-US'
    # TODO: normalize language
    if language and not language.startswith('en'):
        if len(language) > 2:
            lang_tool_lang = language[0:2]
    if lang_tool_lang in _LANG_TOOLS:
        return _LANG_TOOLS[lang_tool_lang]
    try:
        if 'LANGUAGE_TOOL_PORT' in os.environ:
            lang_tool = language_tool_python.LanguageTool(
                language=lang_tool_lang,
                remote_server=f"http://{os.environ.get('LANGUAGE_TOOL_HOST', '127.0.0.1')}:{os.environ['LANGUAGE_TOOL_PORT']}"
            )
        else:
            lang_tool = language_tool_python.LanguageTool(
                language=lang_tool_lang,
                language_tool_download_version=LTP_DOWNLOAD_VERSION
            )
            atexit.register(lambda: lang_tool.close())

        _LANG_TOOLS[lang_tool_lang] = lang_tool
        return lang_tool
    except Exception as e:
        logger.warning("language tool instantiation failure for %s", lang_tool_lang, e)
        return None


def srt_words_to_sentences(words: list[SubRipItem], language: str) -> list[SubRipItem]:
    """
    Collects words into "sentences". There may be multiple sentences per language rules, but we're calling
    sentences a collection of words.
    Use https://pypi.org/project/language-tool-python/ to correct assembled sentences.
    Guidelines:
    https://www.capitalcaptions.com/services/closed-captioning-services/closed-captioning-guidelines/
    - ... maximum 40 characters.
    - Maximum two lines.
    - Adult’s closed caption reading speed set to maximum 250 words per minute/20 characters per second.
    - Children’s closed caption reading speed set to maximum 200 words per minute/17 characters per second.
    - Minimum caption display time 1 second.
    - Maximum caption display time 8 seconds.
    - Caption timings should be set to sync exactly with the start and stop of audio.
    - Where caption timings need to be lengthened to allow for reading speed, extension time should be added after speech finishes and never before it starts.
    """
    chars_per_sentence = 40
    linebreaks_per_sentence = 2
    min_duration_ms = 1000
    max_duration_ms = 8000
    newline = '\n'
    words = list(filter(lambda e: not _is_transcribed_word_suspicious(e), words))
    has_punctuation = any(filter(lambda e: e.text[-1] in ['.', '?', '!'], words))
    has_capitalization = any(filter(lambda e: e.text[0].isupper() and e.text[-1].islower(), words))
    logger.debug(f"has_punctuation = {has_punctuation}, has_capitalization = {has_capitalization}")
    sentences: list[SubRipItem] = []
    spellchecker = get_spell_checker(language)
    s = None
    new_s = True
    for word in words:
        if _is_transcribed_word_suspicious(word, language):
            continue
        if s is not None:
            if has_punctuation:
                if word.text[0].isalpha() and not s.text.strip()[-1].isalpha():
                    new_s = True
            elif word.start.ordinal - s.end.ordinal > SILENCE_FOR_NEW_SENTENCE:
                s.text = _add_punctuation(s.text, language)
                new_s = True

            if word.end.ordinal - s.start.ordinal > max_duration_ms and word.start.ordinal - s.end.ordinal > min_duration_ms:
                s = None
            elif s.end.ordinal - s.start.ordinal > min_duration_ms:
                if len(s.text) + 1 + len(word.text) > chars_per_sentence * linebreaks_per_sentence:
                    s = None

        if s is None:
            if has_capitalization:
                text = word.text
            else:
                if new_s:
                    text = word.text.capitalize()
                else:
                    text = _capitalize(word.text, language, spellchecker)
            new_s = False

            s = SubRipItem(index=len(sentences), start=word.start, end=word.end, text=text)
            sentences.append(s)
        else:
            s.end = word.end
            if has_capitalization:
                word_capitalized = word.text
            else:
                word_capitalized = _capitalize(word.text, language, spellchecker)
            if s.text.endswith(newline):
                s.text = s.text + word_capitalized
            else:
                s.text = s.text + ' ' + word_capitalized

    if s is not None and not has_punctuation:
        s.text = _add_punctuation(s.text, language)

    if not (has_capitalization and has_punctuation):
        lang_tool = _get_lang_tool(language)
        try:
            if lang_tool is not None:
                for sentence in sentences:
                    sentence.text = lang_tool.correct(sentence.text)
        except Exception as e:
            logger.warning("language tool failure", e)

    for sentence in sentences:
        new_text = ""
        new_text_last_line_count = 0
        words = sentence.text.split()
        for word_idx, word in enumerate(words):
            is_last_word = word_idx == len(words) - 1
            if (not is_last_word and new_text_last_line_count > 0
                    and new_text_last_line_count + 1 + len(word) > chars_per_sentence):
                new_text = new_text + newline + word
                new_text_last_line_count = len(word)
            elif new_text:
                new_text = new_text + ' ' + word
                new_text_last_line_count += 1 + len(word)
            else:
                new_text = word
                new_text_last_line_count = len(word)
        sentence.text = new_text

    return sentences


def fix_subtitle_audio_alignment(subtitle_inout: Union[AssFile, SubRipFile], words: SubRipFile, lang='en',
                                 should_add_new_events=True, filename: str = None, input_info: dict = None) -> tuple[
    bool, str]:
    """
    Fix the subtitle to be aligned to the audio using the transcribed words.
    :param subtitle_inout: the subtitle to align, srt or ssa
    :param words: the transcribed words, srt
    :param lang: language
    :param should_add_new_events: True to add new events from unmatched text
    :return: (True if changes were made, alignment metadata value)
    """

    # an offset is defined as the number of seconds between the input subtitle and the matched sentence in 'words'

    # the max number of seconds the 'words' are searched for a matching sentence
    max_offset_ms = 5000
    # minimum fuzz ratios to consider a sentence to match
    min_fuzz_ratios = [88, 85, 80, 70]
    # percentage of words to fuzz +/- from each input subtitle element
    word_count_fuzz_pct = 0.40
    # unclaimed_word_capture_duration_max_ms = 1800
    # multiple runs work, it would be better to fix so a single pass works *shrug
    passes = 2
    # for non-DVR source, passes = 0, so we don't move events but we still capture missing words
    if common.is_ripped_from_media(input_info):
        passes = 0

    subtitle_facade = subtitle.new_subtitle_file_facade(subtitle_inout)

    words_filtered = list(filter(lambda e: not _is_transcribed_word_suspicious(e), words))
    if len(words_filtered) == 0:
        return False, "Skipping subtitle audio alignment due to no valid transcribed words"
    logger.debug("Removed %i suspicious words from transcription", len(words) - len(words_filtered))
    words_start_ms = list(map(lambda e: e.start.ordinal, words_filtered))
    words_start_end_ms = list(map(lambda e: (e.start.ordinal, e.end.ordinal), words_filtered))

    words_silence_start_ms: list[int] = []
    words_silence_start_end_ms: list[tuple[int, int]] = []
    if words_filtered[0].start.ordinal > SILENCE_FOR_SOUND_EFFECT:
        words_silence_start_end_ms.append((0, words_filtered[0].start.ordinal))
        words_silence_start_ms.append(0)
    for words_idx in range(1, len(words_filtered)):
        if words_filtered[words_idx].start.ordinal - words_filtered[
            words_idx - 1].end.ordinal > SILENCE_FOR_SOUND_EFFECT:
            words_silence_start_end_ms.append(
                (words_filtered[words_idx - 1].end.ordinal, words_filtered[words_idx].start.ordinal))
            words_silence_start_ms.append(words_filtered[words_idx - 1].end.ordinal)

    def get_transcription_info(event: subtitle.SubtitleElementFacade) -> Tuple[int, int, str]:
        """
        Get the transcription words and text for an event.
        :param event:
        :return: start word index (inclusive), end word index (inclusive), text assembled from the words
        """
        start_word_idx = find_subtitle_element_idx_ge(words_start_ms, event.start())
        end_word_idx = find_subtitle_element_idx_ge(words_start_ms, event.end())
        while end_word_idx >= start_word_idx and words_start_end_ms[end_word_idx][1] > event.end():
            end_word_idx -= 1
        return start_word_idx, end_word_idx, ' '.join(
            list(map(lambda e: e.text, words_filtered[start_word_idx:end_word_idx + 1])))

    def get_adjustment_stats(use_abs=True):
        adjustments = []
        count = 0
        sane_adjustment_max = 60000
        for event_idx, event in enumerate(events):
            start_adjustment = original_range_ms[event_idx][0] - event.start()
            end_adjustment = original_range_ms[event_idx][1] - event.end()
            if use_abs:
                start_adjustment = abs(start_adjustment)
                end_adjustment = abs(end_adjustment)
            if start_adjustment != 0 or end_adjustment:
                count += 1
            adjustments.append((start_adjustment, end_adjustment))
        max_start_adjustment = max(filter(lambda e: e < sane_adjustment_max, map(lambda e: e[0], adjustments)))
        ave_start_adjustment = average(
            list(filter(lambda e: e < sane_adjustment_max, map(lambda e: e[0], adjustments))))
        max_end_adjustment = max(filter(lambda e: e < sane_adjustment_max, map(lambda e: e[1], adjustments)))
        ave_end_adjustment = average(list(filter(lambda e: e < sane_adjustment_max, map(lambda e: e[1], adjustments))))
        return adjustments, max_start_adjustment, ave_start_adjustment, max_end_adjustment, ave_end_adjustment, count

    def suspicious_last_word(event_idx: int, start_word_idx: int, end_word_idx: int) -> bool:
        if end_word_idx <= start_word_idx:
            return False
        start_ms = words_filtered[start_word_idx].start.ordinal
        last_word = words_filtered[end_word_idx - 1]
        if last_word.end.ordinal - start_ms <= original_duration_ms[event_idx]:
            return False
        if events[event_idx].normalized_texts_endswith(last_word.text):
            return False
        return _is_transcribed_word_ambiguous(last_word.text, lang)

    def suspicious_first_word(event_idx: int, start_word_idx: int, end_word_idx: int) -> bool:
        if end_word_idx <= start_word_idx:
            return False
        first_word = words_filtered[start_word_idx]
        end_ms = words_filtered[end_word_idx - 1].end.ordinal
        if end_ms - first_word.start.ordinal <= original_duration_ms[event_idx]:
            return False
        if events[event_idx].normalized_texts_startswith(first_word.text):
            return False
        return _is_transcribed_word_ambiguous(first_word.text, lang)

    def mark_claimed_words(event: subtitle.SubtitleElementFacade, words_claimed: list[bool],
                           unclaimed_word_events: list):
        try:
            start_word_idx, end_word_idx, _ = get_transcription_info(event)
            for word_idx in range(start_word_idx, end_word_idx + 1):
                words_claimed[word_idx] = True
                try:
                    unclaimed_word_events.remove((word_idx, words_filtered[word_idx]))
                    logger.debug("removed unclaimed word %s", words_filtered[word_idx])
                except ValueError:
                    # may have previously been removed
                    logger.debug("already removed unclaimed word %s", words_filtered[word_idx])
        except ValueError:
            # may not have associated words
            logger.info("get_transcription_info failed for event %i '%s'", event_idx, event.log_text())

    # target ranges that have been found, matches event index, (start from words, end from words) or None
    events = []
    original_range_ms = []
    original_duration_ms = []
    for event_idx, event in subtitle_facade.events():
        event.set_normalized_texts(_subtitle_text_to_plain(event.text()))
        events.append(event)
        original_range_ms.append((event.start(), event.end()))
        original_duration_ms.append(event.duration())
    new_events_count = 0

    if len(events) == 0 or len(words_filtered) == 0:
        return False, ""

    if filename:
        progress_task_name = filename + ' '
    else:
        progress_task_name = ''
    align_progress = progress.progress(f"{progress_task_name}subtitle alignment", 0,
                                       (len(min_fuzz_ratios) + 3) * passes)

    # TODO: original event that is too quiet to be picked up by transcribing
    # TODO: omitted event that transcribing picked up that should be combined with existing event
    last_fuzz_ratio = min_fuzz_ratios[-1]
    for pass_num in range(1, passes + 1):
        found_range_ms: list[Union[None, Tuple[int, int]]] = [None] * len(events)
        words_claimed = [False] * len(words_filtered)
        progress_base = (pass_num - 1) * (len(min_fuzz_ratios) + 4)

        # check if current ranges are good fits
        for event_idx, event in enumerate(events):
            if found_range_ms[event_idx] is not None:
                continue
            try:
                current_start_idx, current_end_idx, current_text = get_transcription_info(event)
                event_texts = event.normalized_texts()
                if any(map(lambda e: len(e) > 3 and fuzz.ratio(e, current_text) >= 96, event_texts)):
                    if not any(words_claimed[current_start_idx:current_end_idx + 1]):
                        logger.debug("current match %i at (%s, %s) for '%s' is '%s'",
                                     event_idx, common.ms_to_ts(event.start()), common.ms_to_ts(event.end()),
                                     event.log_text(), current_text)
                        found_range_ms[event_idx] = (event.start(), event.end())
                        for word_idx in range(current_start_idx, current_end_idx + 1):
                            words_claimed[word_idx] = True
                    else:
                        logger.warning("current NOT matched %i, claimed otherwise at (%s, %s) for '%s' is '%s'",
                                       event_idx, common.ms_to_ts(event.start()), common.ms_to_ts(event.end()),
                                       event.log_text(), current_text)
            except ValueError:
                pass

        # single words match more things erroneously than longer strings of words
        match_attempt: list[list[bool]] = [[]] * len(min_fuzz_ratios)
        for min_fuzz_ratio_idx in range(len(min_fuzz_ratios)):
            match_attempt[min_fuzz_ratio_idx] = [False] * len(events)
        for word_count_min in [8, 5, 3, 2, 1]:
            # run through each ratio and iteratively find matches, using existing matches to narrow the search
            for min_fuzz_ratio_idx, min_fuzz_ratio in enumerate(min_fuzz_ratios):
                _, _, ave_start_adjustment, _, ave_end_adjustment, stats_count = get_adjustment_stats(use_abs=False)
                search_range_adjust_ms = max_offset_ms if ave_start_adjustment == 0 or stats_count < 200 else 2 * abs(
                    ave_start_adjustment)
                for event_idx, event in enumerate(events):
                    if match_attempt[min_fuzz_ratio_idx][event_idx] or found_range_ms[event_idx] is not None:
                        continue

                    if event.is_sound_effect():
                        # let some text matches be made before working with sound effects
                        if word_count_min >= 3 and min_fuzz_ratio_idx < 2:
                            continue
                        # search for gaps in transcription similar to event duration
                        match_attempt[min_fuzz_ratio_idx][event_idx] = True
                        start_search_ms = max(event.start() - max_offset_ms,
                                              max(map(lambda e: e[1],
                                                      filter(lambda f: f is not None, found_range_ms[:event_idx])),
                                                  default=0))
                        end_search_ms = min(event.end() + max_offset_ms,
                                            min(map(lambda e: e[0],
                                                    filter(lambda f: f is not None, found_range_ms[event_idx + 1:])),
                                                default=sys.maxsize))
                        try:
                            start_idx = find_subtitle_element_idx_le(words_silence_start_ms, start_search_ms)
                            end_idx = find_subtitle_element_idx_ge(words_silence_start_ms, end_search_ms)
                        except ValueError:
                            continue
                        silence_candidates: list[tuple[int, int]] = words_silence_start_end_ms[start_idx:end_idx + 1]
                        silence_candidates.sort(key=lambda e: abs(e[0] - (event.start() + ave_start_adjustment)))
                        logger.debug("Matching sound effect from event %i (%s,%s) '%s' in range (%s,%s)",
                                     event_idx, common.ms_to_ts(event.start()), common.ms_to_ts(event.end()),
                                     event.log_text(),
                                     common.ms_to_ts(words_silence_start_ms[start_idx]),
                                     common.ms_to_ts(words_silence_start_ms[end_idx]))
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug("event %i '%s' sound effect candidates %s", event_idx, event.log_text(),
                                         list(map(lambda e: f"({common.ms_to_ts(e[0])},{common.ms_to_ts(e[1])})",
                                                  silence_candidates)))
                        if not silence_candidates:
                            continue

                        if event_idx == 0:
                            silence_start_bound = 0
                        elif found_range_ms[event_idx - 1]:
                            silence_start_bound = found_range_ms[event_idx - 1][1]
                        else:
                            silence_start_bound = event.start() - min(max_offset_ms, abs(ave_start_adjustment) * 1.5)

                        if event_idx >= len(events) - 1:
                            silence_end_bound = sys.maxsize
                        elif found_range_ms[event_idx + 1]:
                            silence_end_bound = found_range_ms[event_idx + 1][0]
                        else:
                            silence_end_bound = event.end() + min(max_offset_ms, abs(ave_end_adjustment) * 1.5)

                        for silence_candidate in silence_candidates:
                            if event.duration() <= 0:
                                continue
                            start_new_ms = max(silence_candidate[0], silence_start_bound)
                            end_new_ms = min(silence_candidate[1], silence_end_bound)
                            silence_duration_pct = (end_new_ms - start_new_ms) / event.duration()
                            if silence_duration_pct < 0.5:
                                continue
                            if silence_duration_pct > 1.0:
                                if start_new_ms == event.start():
                                    end_new_ms = min(end_new_ms, start_new_ms + event.duration())
                                elif end_new_ms == event.end():
                                    start_new_ms = max(start_new_ms, end_new_ms - event.duration())
                                else:
                                    start_new_ms = max(start_new_ms, event.start() + ave_start_adjustment)
                                    end_new_ms = min(end_new_ms, event.end() + ave_end_adjustment)
                                    if start_new_ms >= end_new_ms:
                                        continue

                            logger.debug("event %i sound effect match at (%s %+i, %s %+i) for '%s'", event_idx,
                                         common.ms_to_ts(start_new_ms), start_new_ms - event.start(),
                                         common.ms_to_ts(end_new_ms), end_new_ms - event.end(),
                                         event.log_text())

                            found_range_ms[event_idx] = (start_new_ms, end_new_ms)
                            event.set_start(start_new_ms)
                            event.set_end(end_new_ms)
                            break

                        continue

                    word_count = event.get_normalized_word_count()
                    if word_count < word_count_min:
                        continue

                    event_texts = event.normalized_texts()
                    match_attempt[min_fuzz_ratio_idx][event_idx] = True
                    word_counts = range(floor(word_count * (1.0 - word_count_fuzz_pct)),
                                        ceil(word_count * (1.0 + word_count_fuzz_pct)) + 1)
                    logger.debug("Matching sentence from event %i (%s,%s) '%s', (%i,%i) words",
                                 event_idx, common.ms_to_ts(event.start()), common.ms_to_ts(event.end()), event_texts,
                                 word_counts.start, word_counts.stop)
                    # ignore ranges that have already been matched
                    start_search_ms = max(event.start() - search_range_adjust_ms,
                                          max(map(lambda e: e[1],
                                                  filter(lambda f: f is not None, found_range_ms[:event_idx])),
                                              default=0))
                    end_search_ms = min(event.start() + search_range_adjust_ms,
                                        min(map(lambda e: e[0],
                                                filter(lambda f: f is not None, found_range_ms[event_idx + 1:])),
                                            default=sys.maxsize))
                    try:
                        start_idx = find_subtitle_element_idx_ge(words_start_ms, start_search_ms)
                        end_idx = find_subtitle_element_idx_le(words_start_ms, end_search_ms)
                    except ValueError:
                        continue
                    candidates = dict()
                    for idx in range(start_idx, end_idx + 1):
                        # TODO: handle already masked text: "***", "@#$!", ...
                        logger.debug("Enumerating candidate sentences from word %i %s '%s'", idx,
                                     common.ms_to_ts(words_filtered[idx].start.ordinal), words_filtered[idx].text)
                        for c in word_counts:
                            key = (idx, idx + c)
                            if key[1] > len(words_filtered):
                                continue
                            if key[0] >= key[1]:
                                continue
                            if words_filtered[key[1] - 1].start.ordinal > (
                                    words_filtered[key[0]].start.ordinal + event.duration() + max_offset_ms):
                                continue
                            # check if a gap of silence is included
                            silence_gap_included = False
                            for j in range(key[0], key[1] - 1):
                                if words_filtered[j + 1].start.ordinal - words_filtered[
                                    j].end.ordinal > SILENCE_FOR_NEW_SENTENCE * 2:
                                    logger.debug("Silence gap %i ms between words, skipping candidate",
                                                 words_filtered[j + 1].start.ordinal - words_filtered[j].end.ordinal)
                                    silence_gap_included = True
                            if silence_gap_included:
                                continue
                            try:
                                # ignore ranges that have already been matched
                                words_claimed.index(True, key[0], key[1])
                            except ValueError:
                                value = ' '.join(map(lambda e: e.text, words_filtered[key[0]:key[1]]))
                                candidates[key] = value
                    logger.debug("event %i candidates in range (%s,%s) are %s", event_idx,
                                 common.ms_to_ts(start_search_ms), common.ms_to_ts(end_search_ms),
                                 candidates)
                    if not candidates:
                        logger.debug("no candidates for min_fuzz_ratio %i '%s' (%i,%i)",
                                     min_fuzz_ratio, event_texts, start_search_ms, end_search_ms)
                        continue
                    else:
                        matches = []
                        for event_text in event_texts:
                            matches += (fuzzprocess.extractBests(event_text, candidates, scorer=fuzz.ratio,
                                                                 score_cutoff=last_fuzz_ratio))

                        # remove overlapping matches with lesser ratio
                        if matches:
                            matches.sort(reverse=True, key=lambda e: [e[1], -1 * e[2][0]])
                            # logger.debug("event %i matches before removing overlaps: %s", event_idx, matches)
                            for matches_idx, match in enumerate(matches.copy()):
                                matches_idx2 = matches_idx + 1
                                while matches_idx2 < len(matches):
                                    m = matches[matches_idx2]
                                    if match[2][0] <= m[2][1] and m[2][0] <= match[2][1]:
                                        # overlap
                                        matches.pop(matches_idx2)
                                    else:
                                        matches_idx2 += 1
                            # logger.debug("event %i matches after removing overlaps: %s", event_idx, matches)

                        # check for similar sentences in which the later matches better with the former transcription
                        if len(matches) > 1:
                            matches.sort(key=lambda e: e[2][0])
                            if event_idx > 0 and not found_range_ms[event_idx - 1]:
                                matches.pop(0)
                                logger.debug("event %i multiple transcriptions may match %s, choosing second",
                                             event_idx,
                                             list(map(lambda e: e[1], matches)))
                            else:
                                logger.debug("event %i multiple transcriptions may match %s, choosing earlier",
                                             event_idx,
                                             list(map(lambda e: e[1], matches)))
                        else:
                            matches = list(filter(lambda e: e[1] >= min_fuzz_ratio, matches))

                    if matches:
                        logger.debug("event %i sorted matches: %s", event_idx, matches)
                        match = matches[0]
                        start_new_idx = match[2][0]
                        end_new_idx = match[2][1]

                        # don't be greedy with ambiguous words: 'the', 'the the', etc. causing
                        # action events, i.e. "[ Sighs ]", get clobbered by text events. Keep duration if ending
                        # word(s) do not match original event and are ambiguous words (the, to, ...)
                        while suspicious_last_word(event_idx, start_new_idx, end_new_idx):
                            logger.debug("event %i removing from match suspicious last word %s", event_idx,
                                         words_filtered[end_new_idx - 1].text)
                            end_new_idx -= 1
                        while suspicious_first_word(event_idx, start_new_idx, end_new_idx):
                            logger.debug("event %i removing from match suspicious first word %s", event_idx,
                                         words_filtered[start_new_idx - 1].text)
                            start_new_idx += 1

                        start_new_ms = words_filtered[start_new_idx].start.ordinal
                        end_new_ms = words_filtered[end_new_idx - 1].end.ordinal

                        logger.debug("event %i match at (%s %+i, %s %+i) r%i for '%s' is %s", event_idx,
                                     common.ms_to_ts(start_new_ms), start_new_ms - event.start(),
                                     common.ms_to_ts(end_new_ms), end_new_ms - event.end(),
                                     min_fuzz_ratio, event.log_text(), match)

                        found_range_ms[event_idx] = (start_new_ms, end_new_ms)
                        for word_idx in range(start_new_idx, end_new_idx):
                            words_claimed[word_idx] = True

                        # move event time
                        event.set_start(start_new_ms)
                        event.set_end(end_new_ms)
                    else:
                        logger.debug("not matched r%i for '%s', candidates are %s",
                                     min_fuzz_ratio, event_texts, str(candidates)[:200])
            align_progress.progress(progress_base + min_fuzz_ratio_idx + 1)

        logger.debug("matched count %i/%i", len(list(filter(lambda e: e is not None, found_range_ms))),
                     len(found_range_ms))

        # we've matched all we can, now do something with unclaimed words
        unclaimed_word_events = []
        for word_idx, word in enumerate(words_filtered):
            if not words_claimed[word_idx]:
                unclaimed_word_events.append((word_idx, word))
        adjustments, max_start_adjustment, ave_start_adjustment, max_end_adjustment, ave_end_adjustment, _ = get_adjustment_stats()

        # 1. attach unclaimed words to matched events that are missing duration
        # 2. adjust duration for matched events that have sound effects
        for event_idx, event in enumerate(events):
            if found_range_ms[event_idx] is None:
                continue
            missing_duration = original_duration_ms[event_idx] - event.duration()
            if missing_duration > 600000:
                raise ValueError("missing_duration > 600000")
            if missing_duration <= 0:
                continue
            if event.is_sound_effect():
                continue
            try:
                start_word_idx, end_word_idx, transcribed_text = get_transcription_info(event)
            except ValueError:
                continue
            word_count = event.get_normalized_word_count()
            transcribed_word_count = len(transcribed_text.split())

            event_moved = True
            while event_moved and missing_duration > 0:
                event_moved = False

                new_start_word_idx = None
                new_start_ordinal = None
                beginning_duration = None
                # capture possible missing words
                if (transcribed_word_count < word_count
                        and start_word_idx > 0
                        and not words_claimed[start_word_idx - 1]):
                    beginning_duration = words_filtered[start_word_idx].start.ordinal - words_filtered[
                        start_word_idx - 1].start.ordinal
                    if beginning_duration <= missing_duration:
                        new_start_word_idx = start_word_idx - 1
                        new_start_ordinal = words_filtered[start_word_idx].start.ordinal
                # capture empty space for sound effects
                if new_start_word_idx is None and event.has_beginning_sound_effect():
                    if start_word_idx == 0:
                        beginning_duration = words_filtered[start_word_idx].start.ordinal
                        if beginning_duration <= missing_duration:
                            new_start_ordinal = missing_duration - beginning_duration
                    else:
                        beginning_duration = words_filtered[start_word_idx].start.ordinal - words_filtered[
                            start_word_idx - 1].end.ordinal
                        if 0 < beginning_duration <= missing_duration:
                            new_start_ordinal = words_filtered[start_word_idx - 1].end.ordinal + (
                                    missing_duration - beginning_duration)

                new_end_word_idx = None
                new_end_ordinal = None
                ending_duration = None
                # capture possible missing words
                if (transcribed_word_count < word_count
                        and end_word_idx < len(words_claimed) - 1
                        and not words_claimed[end_word_idx + 1]):
                    ending_duration = words_filtered[end_word_idx + 1].end.ordinal - words_filtered[
                        end_word_idx].end.ordinal
                    if ending_duration <= missing_duration:
                        new_end_word_idx = end_word_idx - 1
                        new_end_ordinal = words_filtered[end_word_idx].end.ordinal
                # capture empty space for sound effects
                if new_end_word_idx is None and event.has_ending_sound_effect():
                    if end_word_idx < len(words_claimed) - 1:
                        ending_duration = words_filtered[end_word_idx + 1].start.ordinal - words_filtered[
                            end_word_idx].end.ordinal
                        if 0 < ending_duration <= missing_duration:
                            new_end_ordinal = words_filtered[end_word_idx + 1].start.ordinal - (
                                    missing_duration - ending_duration)

                # capture the largest missing duration at either beginning or end
                if (new_start_ordinal is not None and 0 <= new_start_ordinal < event.start()
                        and ((
                                     new_end_ordinal is not None and beginning_duration >= ending_duration) or new_end_ordinal is None)):
                    missing_duration -= event.start() - new_start_ordinal
                    event.set_start(new_start_ordinal)
                    event_moved = True
                    logger.debug("event %i moved start (%s%+i,%s%+i)", event_idx,
                                 common.ms_to_ts(event.start()), event.start() - original_range_ms[event_idx][0],
                                 common.ms_to_ts(event.end()), event.end() - original_range_ms[event_idx][1],
                                 )
                    if new_start_word_idx is not None:
                        start_word_idx = new_start_word_idx
                        words_claimed[start_word_idx] = True
                        transcribed_word_count += 1
                elif (new_end_ordinal is not None and new_end_ordinal > event.end()
                      and ((
                                   new_start_ordinal is not None and beginning_duration < ending_duration) or new_start_ordinal is None)):
                    missing_duration -= new_end_ordinal - event.end()
                    event.set_end(new_end_ordinal)
                    event_moved = True
                    logger.debug("event %i moved end (%s%+i,%s%+i)", event_idx,
                                 common.ms_to_ts(event.start()), event.start() - original_range_ms[event_idx][0],
                                 common.ms_to_ts(event.end()), event.end() - original_range_ms[event_idx][1],
                                 )
                    if new_end_word_idx is not None:
                        end_word_idx = new_end_word_idx
                        words_claimed[end_word_idx] = True
                        transcribed_word_count += 1

        align_progress.progress(progress_base + len(min_fuzz_ratios) + 1)

        # 3. move unmatched events based on unclaimed words
        # 4. with no transcription help, move event based on average adjustment

        for event_idx, event in enumerate(events):
            try:
                if event_idx == 0:
                    continue
                matched = found_range_ms[event_idx] is not None
                previous_matched = found_range_ms[event_idx - 1] is not None if event_idx > 0 else False
                event_previous = events[event_idx - 1]
                if not matched:
                    event_moved = False
                    if not event.is_normalized_text_blank():
                        # try to start non-matches at the beginning of an unclaimed word
                        unclaimed_word = next(filter(
                            lambda e: e[1].start.ordinal >= event_previous.end(), unclaimed_word_events), None)
                        if unclaimed_word is not None and len(unclaimed_word[1].text) > 3 and unclaimed_word[
                            1].text in event.normalized_start_words():
                            logger.debug("event %i possible unclaimed word %i '%s'", event_idx,
                                         unclaimed_word[1].index, unclaimed_word[1].text)
                            if event_idx + 1 < len(events):
                                end_limit_ms = events[event_idx + 1].start()
                            else:
                                end_limit_ms = words_filtered[-1].end.ordinal
                            capture_unclaimed_start = max(event_previous.end(),
                                                          unclaimed_word[1].start.ordinal - max(0,
                                                                                                event.duration() - (
                                                                                                        end_limit_ms -
                                                                                                        unclaimed_word[
                                                                                                            1].start.ordinal)))
                            if abs(capture_unclaimed_start - event.start()) < ave_start_adjustment:
                                logger.debug(
                                    "moving event %i '%s' to claim word %i, adjusted to %s %+i", event_idx,
                                    event.log_text(),
                                    unclaimed_word[1].start.ordinal, common.ms_to_ts(capture_unclaimed_start),
                                    capture_unclaimed_start - event.start())
                                event.move(capture_unclaimed_start)
                                words_claimed[unclaimed_word[0]] = True
                                event_moved = True

                    if not event_moved:
                        # no text to match, make it relative to previous event
                        event_previous_adj = event_previous.start() - original_range_ms[event_idx - 1][0]
                        if abs(event_previous_adj) > ave_start_adjustment:
                            event_previous_adj = ave_start_adjustment * (abs(event_previous_adj) / event_previous_adj)
                        if event_idx + 1 < len(original_range_ms) and found_range_ms[event_idx + 1]:
                            event_next_start = events[event_idx + 1].start()
                            event_end_overage = (event.end() + event_previous_adj) - event_next_start
                            if event_end_overage > 0:
                                event_previous_adj -= event_end_overage

                        start_new_ms = max(event_previous.end(), event.start() + event_previous_adj)
                        if start_new_ms == event_previous.end():
                            logger.debug(
                                "moving event %i '%s' immediately after previous event (limited relative move), adjusted to %s %+i",
                                event_idx,
                                event.log_text(),
                                common.ms_to_ts(event_previous.end()), event_previous.end() - event.start())
                        elif start_new_ms != event.start():
                            logger.debug(
                                "moving event %i '%s' relative to previous event, adjusted to %s %+i",
                                event_idx,
                                event.log_text(),
                                common.ms_to_ts(start_new_ms), start_new_ms - event.start())
                        event.move(start_new_ms)

                # fix overlaps
                if event_previous.start() > event.start():
                    logger.debug("event %i start overlaps %i start, %s > %s, matched? %r, previous matched? %r",
                                 event_idx - 1, event_idx,
                                 common.ms_to_ts(event_previous.start()), common.ms_to_ts(event.start()),
                                 matched, previous_matched)
                    if previous_matched:
                        event.move(event_previous.end() + 1)
                    elif event_idx > 1:
                        event_previous.move_end(event.start() - 1)
                        if event_idx > 2 and event_previous.start() < events[
                            event_idx - 2].end() < event_previous.end():
                            event_previous.set_start(events[event_idx - 2].end() + 1)
                if event_previous.end() > event.start():
                    logger.debug("event %i end overlaps %i start, %s > %s, matched? %r, previous matched? %r",
                                 event_idx - 1, event_idx,
                                 common.ms_to_ts(event_previous.end()), common.ms_to_ts(event.start()),
                                 matched, previous_matched)
                    if previous_matched:
                        event.move(event_previous.end() + 1)
                    else:
                        event_previous.set_end(event.start() - 1)

                mark_claimed_words(event, words_claimed, unclaimed_word_events)
                mark_claimed_words(event_previous, words_claimed, unclaimed_word_events)
            finally:
                event_previous = event

        align_progress.progress(progress_base + len(min_fuzz_ratios) + 2)

        # extend durations based on original, not to overlap
        for event_idx, event in enumerate(events[:-1]):
            missing_duration = original_duration_ms[event_idx] - event.duration()
            if missing_duration > 0:
                # we've already tried to capture words, now fill in space at end and then beginning
                end_space = events[event_idx + 1].start() - event.end()
                if event_idx > 0:
                    start_space = event.start() - events[event_idx - 1].end()
                else:
                    start_space = 0

                if end_space > 0:
                    end_pad = min(missing_duration, end_space)
                    event.set_end(event.end() + end_pad)
                    missing_duration -= end_pad
                    logger.debug("event %i '%s' padded end %+i", event_idx, event.log_text(), end_pad)
                if missing_duration > 0 and start_space > 0:
                    start_pad = min(missing_duration, start_space)
                    event.set_start(event.start() - start_pad)
                    missing_duration -= start_pad
                    logger.debug("event %i '%s' padded start %+i", event_idx, event.log_text(), start_pad)
            # expand based on original range
            if original_range_ms[event_idx][0] < event.start():
                if event_idx == 0:
                    event.set_start(original_range_ms[event_idx][0])
                    logger.debug("event %i '%s' extended up to original start %s",
                                 event_idx, event.log_text(), common.ms_to_ts(event.start()))
                elif events[event_idx - 1].end() < original_range_ms[event_idx][0]:
                    event.set_start(max(events[event_idx - 1].end(), original_range_ms[event_idx][0]))
                    logger.debug("event %i '%s' extended up to original start %s",
                                 event_idx, event.log_text(), common.ms_to_ts(event.start()))
            if original_range_ms[event_idx][1] > event.end():
                if events[event_idx + 1].start() > original_range_ms[event_idx][1]:
                    event.set_end(min(events[event_idx + 1].start(), original_range_ms[event_idx][1]))
                    logger.debug("event %i '%s' extended up to original end %s",
                                 event_idx, event.log_text(), common.ms_to_ts(event.end()))

        align_progress.progress(progress_base + len(min_fuzz_ratios) + 3)

        # report
        last_word_idx = -1
        for event_idx, event in enumerate(events):
            try:
                start_word_idx = find_subtitle_element_idx_ge(words_start_ms, event.start())
                end_word_idx = find_subtitle_element_idx_ge(words_start_ms, event.end())
            except ValueError:
                continue
            while end_word_idx >= start_word_idx and words_start_end_ms[end_word_idx][1] > event.end():
                end_word_idx -= 1

            if 0 <= last_word_idx < (start_word_idx - 1):
                unclaimed_words = []
                unclaimed_start = sys.maxsize
                unclaimed_end = 0
                for i in range(last_word_idx, start_word_idx):
                    if not words_claimed[i]:
                        unclaimed_words.append(words_filtered[i].text)
                        unclaimed_start = min(unclaimed_start, words_filtered[i].start.ordinal)
                        unclaimed_end = max(unclaimed_end, words_filtered[i].end.ordinal)
                unclaimed_text = ' '.join(unclaimed_words)
                if unclaimed_start < unclaimed_end:
                    logger.debug("unclaimed words (%i,%i) '%s'", unclaimed_start, unclaimed_end, unclaimed_text)
            last_word_idx = end_word_idx

            transcribed_text = ' '.join(
                list(map(lambda e: e.text, words_filtered[start_word_idx:end_word_idx + 1])))
            ratio = average(list(map(lambda e: fuzz.ratio(e, transcribed_text), event.normalized_texts())))
            matches = "matches" if found_range_ms[event_idx] is not None else "claims"
            logger.log(logging.WARNING if ratio < 50 else logging.DEBUG,
                       "event %i (%s%+i - %s%+i %i ms was %i ms) '%s' %s words '%s' with ratio %i",
                       event_idx,
                       common.ms_to_ts(event.start()), event.start() - original_range_ms[event_idx][0],
                       common.ms_to_ts(event.end()), event.end() - original_range_ms[event_idx][1],
                       event.duration(), original_duration_ms[event_idx],
                       event.log_text(), matches, transcribed_text, ratio)

    if should_add_new_events:
        last_word_idx = -1
        event_idx = 0
        while event_idx < len(events):
            event = events[event_idx]
            try:
                start_word_idx, end_word_idx, _ = get_transcription_info(event)
            except ValueError:
                event_idx += 1
                continue

            if 0 <= last_word_idx < (start_word_idx - 1):
                unclaimed_words = []
                for i in range(last_word_idx, start_word_idx):
                    word = words_filtered[i]
                    if word.start.ordinal >= events[event_idx - 1].end() and word.end.ordinal <= event.start():
                        unclaimed_words.append(word)
                        end_word_idx = i
                if len(unclaimed_words) > 0:
                    logger.debug("creating events for unclaimed words '%s'",
                                 list(map(lambda e: e.text, unclaimed_words)))
                    for sentence in srt_words_to_sentences(unclaimed_words, lang):
                        event = subtitle_facade.insert(event_idx)
                        event.set_start(sentence.start.ordinal)
                        event.set_end(sentence.end.ordinal)
                        event.set_text(sentence.text)
                        event.set_normalized_texts(_subtitle_text_to_plain(sentence.text))
                        events.insert(event_idx, event)
                        logger.debug("inserted new event at %i (%s,%s) '%s'", event_idx,
                                     common.ms_to_ts(event.start()), common.ms_to_ts(event.end()), event.log_text())
                        new_events_count += 1
                        original_range_ms.insert(event_idx, (sentence.start.ordinal, sentence.end.ordinal))
                        original_duration_ms.insert(event_idx, sentence.end.ordinal - sentence.start.ordinal)
                        event_idx += 1

            event_idx += 1
            last_word_idx = end_word_idx + 1

        # ensure monotonically increasing indices
        for event_idx, event in enumerate(events):
            event.set_index(event_idx + 1)

    # check stats for material changes
    # use these stats to determine if we adjusted enough to make a difference and return False if not
    adjustments, max_start_adjustment, ave_start_adjustment, max_end_adjustment, ave_end_adjustment, _ = get_adjustment_stats()
    logger.info(
        "subtitle alignment stats: max_start_adjustment %i, max_end_adjustment %i, ave_start_adjustment %i, ave_end_adjustment %i, new events %i",
        max_start_adjustment, max_end_adjustment, ave_start_adjustment, ave_end_adjustment, new_events_count)
    stats_str = f"max_start_adj {max_start_adjustment:.1f}ms, max_end_adj {max_end_adjustment:.1f}ms, ave_start_adj {ave_start_adjustment:.1f}ms, ave_end_adj {ave_end_adjustment:.1f}ms, new_events {new_events_count}"
    for event_idx, adjustment in enumerate(adjustments):
        adjustment_log_level = 0
        if adjustment[0] > max_start_adjustment / 2 or adjustment[1] > max_end_adjustment / 2:
            adjustment_log_level = logging.WARNING
        if (events[event_idx].start() >= events[event_idx].end()
                or (event_idx > 0 and events[event_idx - 1].end() > events[event_idx].start())):
            adjustment_log_level = logging.ERROR
        if adjustment_log_level > 0:
            logger.log(adjustment_log_level,
                       "subtitle alignment stats: event %i adjustment (%i,%i) (%s - %s) '%s'",
                       event_idx,
                       events[event_idx].start() - original_range_ms[event_idx][0],
                       events[event_idx].end() - original_range_ms[event_idx][1],
                       common.ms_to_ts(events[event_idx].start()), common.ms_to_ts(events[event_idx].end()),
                       events[event_idx].log_text())

    changed = ave_start_adjustment > 100 or ave_end_adjustment > 100 or new_events_count > 0

    align_progress.stop()

    return changed, stats_str


def create_subtitle_idx_file(subtitle_idx_filename: str, width: int, height: int):
    with open(subtitle_idx_filename, 'wt') as idx_file:
        # https://wiki.multimedia.cx/index.php?title=VOBsub
        idx_file.write(f"""# VobSub index file, v7 (do not modify this line!)
size: {width}x{height}
palette: 000000, 828282, 828282, 828282, 828282, 828282, 828282, ffffff, 828282, bababa, 828282, 828282, 828282, 828282, 828282, 828282
custom colors: ON, tridx: 1000, colors: 000000, ffffff, 000000, 000000
""")


def profanity_filter_cli(argv) -> int:
    global debug

    no_curses = False
    dry_run = False
    keep = False
    force = False
    filter_skip = None
    mark_skip = None
    unmark_skip = None
    language = constants.LANGUAGE_ENGLISH
    workdir = config.get_work_dir()
    mute_channels = None

    try:
        opts, args = getopt.getopt(
            list(argv), "nkdrf",
            ["dry-run", "keep", "debug", "remove", "mark-skip", "unmark-skip", "force", "work-dir=",
             "mute-voice-channels", "mute-all-channels", "verbose", "no-curses"])
    except getopt.GetoptError:
        usage()
        return CMD_RESULT_ERROR
    for opt, arg in opts:
        if opt == '--help':
            usage()
            return CMD_RESULT_ERROR
        elif opt in ("-n", "--dry-run"):
            dry_run = True
            no_curses = True
        elif opt in ("-k", "--keep"):
            keep = True
        elif opt in ("-d", "--debug"):
            debug = True
        elif opt == "--verbose":
            logging.root.setLevel(logging.DEBUG)
        elif opt == "--no-curses":
            no_curses = True
        elif opt in ("-r", "--remove"):
            filter_skip = True
        elif opt == "--mark-skip":
            mark_skip = True
        elif opt == "--unmark-skip":
            unmark_skip = True
        elif opt in ("-f", "--force"):
            force = True
        elif opt == "--work-dir":
            workdir = arg
        elif opt == "--mute-voice-channels":
            mute_channels = config.MuteChannels.VOICE
        elif opt == "--mute-all-channels":
            mute_channels = config.MuteChannels.ALL

    if len(args) == 0:
        usage()
        return CMD_RESULT_ERROR

    atexit.register(common.finish)

    return common.cli_wrapper(
        profanity_filter, *args, dry_run=dry_run, keep=keep, force=force, filter_skip=filter_skip,
        mark_skip=mark_skip, unmark_skip=unmark_skip, workdir=workdir, verbose=True,
        mute_channels=mute_channels, language=language, no_curses=no_curses)


if __name__ == '__main__':
    sys.exit(profanity_filter_cli(sys.argv[1:]))
