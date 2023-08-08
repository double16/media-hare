#!/usr/bin/env python3

import atexit
import copy
import getopt
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from bisect import bisect_left, bisect_right
from math import ceil, floor
from pathlib import Path
from typing import Tuple, Union

import pysrt
from ass_parser import read_ass, write_ass, AssFile, AssEventList, CorruptAssLineError
from numpy import loadtxt, average
from pysrt import SubRipItem, SubRipFile, SubRipTime
from thefuzz import fuzz
from thefuzz import process as fuzzprocess
from num2words import num2words

import common
from common import subtitle, tools, config, constants, progress, edl_util

# Increment when a coding change materially effects the output
FILTER_VERSION = 12
AUDIO_TO_TEXT_VERSION = 3

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
""", file=sys.stderr)


def profanity_filter(*args, **kwargs) -> int:
    try:
        return do_profanity_filter(*args, **kwargs)
    except CorruptAssLineError:
        logger.error("Corrupt ASS subtitle in %s", args[0])
        return CMD_RESULT_ERROR
    finally:
        common.finish()


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
    logger.info("current filter hash = %s, current filter version = %s, current audio-to-text version = %s",
                current_filter_hash, current_filter_version, current_audio2text_version)

    if not force:
        if current_filter_version and int(current_filter_version) > FILTER_VERSION:
            logger.info("Future filter version found: %s", current_filter_version)
            return CMD_RESULT_UNCHANGED
        if current_audio2text_version and int(current_audio2text_version) > AUDIO_TO_TEXT_VERSION:
            logger.info("Future audio2text version found: %s", current_audio2text_version)
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
                FILTER_VERSION) and current_audio2text_version in [None, '', str(AUDIO_TO_TEXT_VERSION)]:
            logger.info("Stream is already filtered")
            return CMD_RESULT_UNCHANGED

    if not subtitle_original and filter_skip:
        if mark_skip:
            return _tag_as_skipped(filename, tags_filename, input_info, dry_run=dry_run, debug=debug, verbose=verbose)
        return CMD_RESULT_UNCHANGED

    subtitle_srt_generated = None
    subtitle_srt_words = None
    audio_to_text_version = current_audio2text_version
    audio_to_text_subtitle_version = (subtitle_original or {}).get(constants.K_TAGS, {}).get(
        constants.K_AUDIO_TO_TEXT_VERSION, '')
    audio_to_text_filter = (subtitle_original or {}).get(constants.K_TAGS, {}).get(constants.K_AUDIO_TO_TEXT_FILTER, '')

    if need_original_subtitle_ocr(subtitle_original=subtitle_original,
                                  media_duration=float(input_info[constants.K_FORMAT][constants.K_DURATION]),
                                  force=force):
        subtitle_srt_generated = ocr_subtitle_bitmap_to_srt(input_info, temp_base, language, verbose=verbose)

    if audio_original:
        if detect_transcribed_by_version_3(current_audio2text_version, input_info, subtitle_original):
            subtitle_original[constants.K_TAGS][constants.K_AUDIO_TO_TEXT_VERSION] = "3"
            audio_to_text_subtitle_version = "3"

        need_original, need_words = need_original_subtitle_transcribed(subtitle_original=subtitle_original,
                                                                       subtitle_words=subtitle_words,
                                                                       current_audio2text_version=current_audio2text_version,
                                                                       media_duration=float(
                                                                           input_info[constants.K_FORMAT][
                                                                               constants.K_DURATION]),
                                                                       force=force)
        if need_original or need_words:
            # We may only need the words from transcription
            if need_original:
                logger.info("%s Transcribing for text and words", base_filename)
            else:
                logger.info("%s Transcribing for words", base_filename)
            audio_channels = int(audio_original.get(constants.K_CHANNELS, 0))
            if audio_channels > 2:
                audio_to_text_filter = 'pan=1c|FC<0.3*FL+FC+0.3*FR,anlmdn'
            else:
                audio_to_text_filter = 'anlmdn'
            _srt_text, subtitle_srt_words = audio_to_srt(input_info, audio_original, workdir, audio_to_text_filter,
                                                         language, verbose=verbose)
            if _srt_text and os.stat(_srt_text).st_size > 0:
                audio_to_text_version = AUDIO_TO_TEXT_VERSION
                if (need_original or not subtitle_original) and subtitle_srt_generated is None:
                    audio_to_text_subtitle_version = AUDIO_TO_TEXT_VERSION
                    subtitle_srt_generated = _srt_text

    tags = input_info[constants.K_FORMAT].get(constants.K_TAGS, {}).copy()
    tags[constants.K_FILTER_HASH] = filter_hash
    tags[constants.K_FILTER_VERSION] = FILTER_VERSION
    tags[constants.K_AUDIO_TO_TEXT_VERSION] = audio_to_text_version if audio_to_text_version else ''
    tags[constants.K_AUDIO_TO_TEXT_FILTER] = audio_to_text_filter if audio_to_text_filter else ''
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
        logger.debug(f"Removing filtered streams")

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
                              f'{constants.K_AUDIO_TO_TEXT_VERSION}={audio_to_text_subtitle_version if audio_to_text_subtitle_version else ""}',
                              "-disposition:s:0", "default"])
            subtitle_output_idx = 1
        else:
            subtitle_output_idx = 0
        if subtitle_srt_words is not None:
            arguments.extend(["-i", subtitle_srt_words,
                              "-map", f"{streams_file + 2}:0",
                              f"-metadata:s:s:1", f'title={constants.TITLE_WORDS}',
                              "-metadata:s:s:1",
                              f'{constants.K_AUDIO_TO_TEXT_VERSION}={audio_to_text_version if audio_to_text_version else ""}',
                              f"-disposition:s:1", "-default+metadata"])
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
            ass_data = read_ass(Path(subtitle_original_filename))
            if subtitle_words_filename:
                fix_subtitle_audio_alignment(ass_data, pysrt.open(subtitle_words_filename), lang=language)
                if debug:
                    write_ass(ass_data, Path(f"{debug_base}.aligned.{subtitle_codec}"))
                    shutil.copy(subtitle_original_filename, f"{debug_base}.original.{subtitle_codec}")
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
            if subtitle_words_filename:
                fix_subtitle_audio_alignment(srt_data, pysrt.open(subtitle_words_filename), lang=language)
                if debug:
                    srt_data.save(Path(f"{debug_base}.aligned.{subtitle_codec}"), 'utf-8')
                    shutil.copy(subtitle_original_filename, f"{debug_base}.original.{subtitle_codec}")
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
            ["-metadata", f"{constants.K_AUDIO_TO_TEXT_FILTER}={audio_to_text_filter if audio_to_text_filter else ''}"])
        if len(stopped_spans) > 0:
            arguments.extend(["-metadata", f"{constants.K_FILTER_STOPPED}={span_list_to_str(stopped_spans)}"])
        arguments.extend(["-metadata", f"{constants.K_FILTER_SKIP}="])
        if common.should_replace_media_title(input_info):
            arguments.extend(
                ['-metadata', f"{constants.K_MEDIA_TITLE}={common.get_media_title_from_filename(input_info)}"])
        arguments.extend(['-metadata', f"{constants.K_MEDIA_PROCESSOR}={constants.V_MEDIA_PROCESSOR}"])
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
                f'{constants.K_AUDIO_TO_TEXT_VERSION}={audio_to_text_subtitle_version if audio_to_text_subtitle_version else ""}'
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
        arguments.extend(["-map", f"{streams_file}:{extra_audio[constants.K_STREAM_INDEX]}",
                          f"-c:a:{audio_output_idx}", "copy",
                          f"-disposition:a:{audio_output_idx}", "0"])
        audio_output_idx += 1

    # Remaining subtitle streams
    for extra_audio in list(filter(lambda stream: stream[constants.K_CODEC_TYPE] == constants.CODEC_SUBTITLE and stream[
        constants.K_STREAM_INDEX] not in [subtitle_original_idx, subtitle_filtered_idx, subtitle_filtered_forced_idx,
                                          subtitle_words_idx],
                                   input_info['streams'])):
        arguments.extend(["-map", f"{streams_file}:{extra_audio[constants.K_STREAM_INDEX]}",
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

    common.match_owner_and_perm(target_path=temp_filename, source_path=filename)

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
    result = list(map(lambda e: phrase_to_pattern(e), result))
    result = list(map(lambda e: re.compile(e, flags=re.IGNORECASE), result))
    return result


def load_allow_list() -> list[re.Pattern]:
    result = loadtxt(os.path.join(os.path.dirname(common.__file__), 'allow_list.txt'), dtype='str', delimiter='\xFF')
    result = list(filter(phrase_list_accept_condition, result))
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
    except:
        pass

    return force


def need_original_subtitle_transcribed(subtitle_original: dict, subtitle_words: dict, current_audio2text_version: str,
                                       media_duration: float, force: bool) -> Tuple[bool, bool]:
    """
    Determine if the original subtitle needs transcribed from audio.
    :param subtitle_original:
    :param current_audio2text_version:
    :return: True for original needed, True for words needed
    """
    need_original = False
    need_words = False

    if not subtitle_words:
        need_words = True

    if subtitle_original:
        duration_s = subtitle_original.get('tags', {}).get('DURATION', '')
        try:
            if duration_s:
                duration = edl_util.parse_edl_ts(duration_s)
                if duration < (media_duration * 0.60):
                    need_original = True
        except:
            pass
    else:
        need_original = True
        subtitle_original = {}

    if force and current_audio2text_version not in [None, '']:
        need_original = need_original or subtitle_original.get('tags', {}).get(constants.K_AUDIO_TO_TEXT_VERSION,
                                                                               '') not in [None, '']
        need_words = True
    if current_audio2text_version not in [None, '', str(AUDIO_TO_TEXT_VERSION)]:
        need_original = need_original or subtitle_original.get('tags', {}).get(constants.K_AUDIO_TO_TEXT_VERSION,
                                                                               '') not in [None, '',
                                                                                           str(AUDIO_TO_TEXT_VERSION)]
        need_words = True

    return need_original, need_words


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
        subtitle_original_arguments = ['-nostdin', "-loglevel", "error",
                                       '-i', input_info['format']['filename'],
                                       '-map', f'0:{subtitle_original[constants.K_STREAM_INDEX]}',
                                       '-c', 'copy',
                                       '-f', 'srt', '-']
        logger.info('ffmpeg %s', ' '.join(subtitle_original_arguments))
        subtitle_original_proc = tools.ffmpeg.Popen(subtitle_original_arguments, stdout=subprocess.PIPE,
                                                    stderr=subprocess.DEVNULL, bufsize=1, text=True)
        transcribed = True
        for sub in pysrt.stream(subtitle_original_proc.stdout):
            if PATTERN_DETECT_TRANSCRIBED_BY_VERSION_3.search(sub.text):
                transcribed = False
                break
        if transcribed:
            logger.info("Detected original subtitle was transcribed in version 3")
        subtitle_original_proc.stdout.close()
        subtitle_original_proc.wait()
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

    if subtitle_filename is None:
        return None

    tools.subtitle_edit.run([subtitle_filename], check=True)
    if not os.access(subtitle_srt_filename, os.R_OK):
        logger.error(f"SRT not generated from OCR")
        return None

    word_found_pct = words_in_dictionary_pct(subtitle_srt_filename, language,
                                             float(input_info[constants.K_FORMAT][constants.K_DURATION]))
    if word_found_pct < WORD_FOUND_PCT_THRESHOLD:
        logger.error(f"OCR text appears to be incorrect, {word_found_pct}% words found in {language} dictionary")
        return None

    # TODO: Add "[ OCR by media-hare+SubtitleEdit ]" at beginning and end

    return subtitle_srt_filename


def audio_to_srt(input_info: dict, audio_original: dict, workdir, audio_filter: str = None, language=None,
                 verbose=False) -> Union[Tuple[str, str], Tuple[None, None]]:
    """
    Attempts to create a text subtitle from the original audio stream.
    1. vosk does not seem to like filenames with spaces, it's thrown a division by zero
    2. audio stream is being converted to AC3 stereo with default ffmpeg bitrate (192kbps) for most compatibility
    3. --tasks is being set but so far it doesn't seem to yield more cores used
    :return: (srt filename for subtitle, srt filename for words) or (None, None)
    """
    global debug

    try:
        from vosk import Model, KaldiRecognizer, GpuInit, GpuThreadInit, SetLogLevel
        GpuInit()
        GpuThreadInit()
        SetLogLevel(-99)
    except ImportError as e:
        logger.warning("Cannot transcribe audio, vosk missing")
        return None, None

    freq = 16000
    chunk_size = 4000

    temp_fd, subtitle_srt_filename = tempfile.mkstemp(dir=workdir, suffix='.srt')
    os.close(temp_fd)
    if not debug:
        common.TEMPFILENAMES.append(subtitle_srt_filename)

    temp_fd, words_filename = tempfile.mkstemp(dir=workdir, suffix='.srt')
    os.close(temp_fd)
    if not debug:
        common.TEMPFILENAMES.append(words_filename)

    # Converting to ac3 isn't necessary as vosk is using ffmpeg to convert the audio as follows. However, vosk seems to
    # have trouble with spaces in file names so we'd need to still extract to a temp name or symlink.
    # ffmpeg -nostdin -loglevel quiet -i /tmp/tmp09gxo8oz.ac3 -ar 16000.0 -ac 1 -f s16le -
    extract_command = ['-nostdin', "-loglevel", "error",
                       '-i', input_info['format']['filename'],
                       '-map', f'0:{audio_original[constants.K_STREAM_INDEX]}',
                       '-ar', str(freq)]
    if audio_filter:
        if 'pan=' not in audio_filter:
            extract_command.extend(['-ac', '1'])
        extract_command.extend(['-af', audio_filter])
    else:
        extract_command.extend(['-ac', '1'])
    extract_command.extend(['-f', 's16le', '-'])

    if verbose:
        logger.info(tools.ffmpeg.array_as_command(extract_command))
    audio_process = tools.ffmpeg.Popen(extract_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    _vosk_language = vosk_language(language)
    model = Model(model_name=vosk_model(_vosk_language), lang=_vosk_language)
    rec = KaldiRecognizer(model, freq)
    rec.SetWords(True)

    audio_process.stdout.read(44)  # skip header
    results = []
    audio_progress = progress.progress(f"{os.path.basename(input_info['format']['filename'])} transcription", 0,
                                       int(float(
                                           input_info[constants.K_FORMAT][constants.K_DURATION])))
    while True:
        data = audio_process.stdout.read(chunk_size)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if 'result' in result:
                results.append(result)
                if len(result['result']) > 0:
                    audio_progress.progress(result['result'][-1]['end'])
                if 'text' in result:
                    logger.debug("text: %s", result['text'])
    result = json.loads(rec.FinalResult())
    if 'result' in result:
        results.append(result)
        if 'text' in result:
            logger.debug("text: %s", result['text'])
    audio_process.stdout.close()
    audio_progress.stop()
    extract_return_code = audio_process.wait()
    if extract_return_code != 0:
        logger.error("Cannot transcribe audio, ffmpeg returned %s", extract_return_code)
        return None, None

    subs_words = []
    for i, res in enumerate(results):
        words = res['result']
        for word in words:
            s = SubRipItem(index=len(subs_words),
                           start=SubRipTime(seconds=word['start']),
                           end=SubRipTime(seconds=word['end']),
                           text=word['word'])
            subs_words.append(s)

    srt_words = SubRipFile(items=subs_words, path=words_filename)
    srt_words.save(Path(words_filename), 'utf-8')

    subs_text = srt_words_to_sentences(subs_words)
    srt = SubRipFile(items=subs_text, path=subtitle_srt_filename)
    srt.save(Path(subtitle_srt_filename), 'utf-8')

    word_found_pct = words_in_dictionary_pct(subtitle_srt_filename, language,
                                             float(input_info[constants.K_FORMAT][constants.K_DURATION]))
    if word_found_pct < WORD_FOUND_PCT_THRESHOLD:
        logger.error(
            f"audio-to-text transcription appears to be incorrect, {word_found_pct}% words found in {language} dictionary")
        return None, None

    # TODO: Add "[ transcribed by media-hare+Vosk ]" at beginning and end

    return subtitle_srt_filename, words_filename


PATTERN_WORDS_IN_DICT_SPLIT = re.compile('[^A-Za-z\' ]+')


def words_in_dictionary_pct(subtitle_srt_filename, language: str, duration: float):
    try:
        import hunspell

        # verify with spell checker (hunspell) that text looks like English
        if language == constants.LANGUAGE_ENGLISH:
            hobj = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
        else:
            hobj = hunspell.HunSpell(f'/usr/share/hunspell/{language[0:2]}.dic',
                                     f'/usr/share/hunspell/{language[0:2]}.aff')
        word_count = 0
        word_found_count = 0
        srt_data = pysrt.open(subtitle_srt_filename)
        for event in list(srt_data):
            for word in PATTERN_WORDS_IN_DICT_SPLIT.sub(' ', event.text).split():
                word_count += 1
                if hobj.spell(word):
                    word_found_count += 1
        if word_count < 100 and duration > 630:
            logger.warning(f"word count less than 100 for duration {duration}, returning 0%")
            return 0.0
        word_found_pct = 100.0 * float(word_found_count) / (float(word_count) + 0.001)
        logger.info(f"SRT words = {word_count}, found = {word_found_count}, {word_found_pct}%")
        return word_found_pct
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning("hunspell not found, skipping dictionary check")
        return 100.0


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
    return ((begin, end), original[begin:end])


PHRASE_FANCY_WORD_BREAKS = re.compile(r'([^\s^$]+)\s*', flags=re.IGNORECASE)
PHRASE_BEGINNING_MARKER = re.compile(r'[\^]')
PHRASE_STARTING_WORD_BREAK = re.compile(r'.')


def phrase_to_pattern(phrase):
    """
    Transforms a phrase, which is roughly a regex, into something that will match both SRT and ASS markup.
    """
    phrase_fancy_word_breaks = PHRASE_FANCY_WORD_BREAKS.sub(r'(\1)' + WORD_BREAKS, phrase)
    phrase_beginning_marker = PHRASE_BEGINNING_MARKER.sub(NO_PREVIOUS_WORD, phrase_fancy_word_breaks)
    phrase_starting_word_break = PHRASE_STARTING_WORD_BREAK.sub(WORD_BREAKS, '.') + phrase_beginning_marker
    return phrase_starting_word_break


def phrase_list_accept_condition(e):
    e = e.strip()
    if len(e) == 0:
        return False
    return e[0] != '#'


def vosk_language(language: str) -> str:
    """
    Get the language code for the Vosk transcriber from the three character language code.
    :param language: three character language code
    :returns: language code appropriate for Vosk
    """
    if language == 'spa':
        return 'es'
    if language in ['eng', 'en']:
        return 'en-us'
    return language[0:2]


def vosk_model(language: str) -> Union[None, str]:
    """
    Get the Vosk transcriber model to use from the three character language code.
    :param language: three character language code
    :returns: model name or None to let Vosk choose a model based on the language
    """
    if language == 'en-us':
        return 'vosk-model-en-us-0.22-lgraph'
    elif language == 'es':
        return 'vosk-model-small-es-0.22'
    return None


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


def find_subtitle_element_idx_ge(time_ordinals: list[int], start: float) -> int:
    """
    Find leftmost element greater than or equal to x
    https://docs.python.org/3/library/bisect.html
    """
    i = bisect_left(time_ordinals, start)
    if i != len(time_ordinals):
        return i
    raise ValueError


def find_subtitle_element_idx_le(time_ordinals: list[int], start: float) -> int:
    """
    Find rightmost element less than or equal to x
    https://docs.python.org/3/library/bisect.html
    """
    i = bisect_right(time_ordinals, start)
    if i:
        return i - 1
    raise ValueError


SUBTITLE_TEXT_TO_PLAIN_REMOVE = re.compile(r"\[.*?]|\(.*?\)|\{.*?}|<.*?>")
SUBTITLE_TEXT_TO_PLAIN_WS = re.compile(r"\\[A-Za-z]|[,.?$!*&\"-]|\b'|'\b|[\u007F-\uFFFF]")
SUBTITLE_TEXT_TO_PLAIN_SQUEEZE_WS = re.compile(r"\s+")
SUBTITLE_TEXT_TO_PLAIN_NUMBERS = re.compile(r"(\d+(?:[\d.,]+\d)?)")
SUBTITLE_TEXT_TO_PLAIN_ORDINALS = re.compile(r"(\d+)(?:st|nd|rd|th)")
SUBTITLE_TEXT_TO_PLAIN_YEAR = re.compile(r"(\d+\d+\d+\d+)s?")
SUBTITLE_TEXT_TO_PLAIN_CURRENCY = re.compile(r"\$(\d+(?:[\d.,]+\d)?)")
SUBTITLE_TEXT_TO_PLAIN_ABBREV_DICT = {
    'dr.': 'doctor',
    'mr.': 'mister',
    'mrs.': 'misses',
}
SUBTITLE_TEXT_TO_PLAIN_ABBREV_PATTERN = re.compile(
    "(" + "|".join(map(lambda e: e.replace('.', '[.]'), SUBTITLE_TEXT_TO_PLAIN_ABBREV_DICT.keys())) + ")",
    flags=re.IGNORECASE)


def _num2words_cardinal(numbers: str, lang: str) -> str:
    if numbers.isdigit() and len(numbers) > 3:
        # use a word per digit, such as a spoken account number
        digits = [num2words(ch, lang=lang) for ch in numbers]
        return ' '.join(digits)

    return num2words(numbers.replace(',', ''), lang=lang)


def _subtitle_text_to_plain(text: str, lang='en') -> str:
    """
    Clean up subtitle text to remove non-spoken markers
    remove [ Name ]
    remove ( Name )
    remove { ASS tags }
    replace "\\N" with " "
    remove '.' from abbreviations
    all lowercase to match closer to transcription
    squeeze whitespace to match closer to transcription
    numbers to text?
    """
    # TODO: Some USD like $12.99 is being represented as euros
    # TODO: Handle strings like phone numbers: (800) 555-1212
    # TODO: Match numbers like "622" = "six twenty two"
    text = SUBTITLE_TEXT_TO_PLAIN_CURRENCY.sub(lambda n: num2words(n.group(1), to='currency', lang=lang), text)
    text = SUBTITLE_TEXT_TO_PLAIN_ORDINALS.sub(lambda n: num2words(n.group(1), to='ordinal', lang=lang), text)
    text = SUBTITLE_TEXT_TO_PLAIN_YEAR.sub(lambda n: num2words(n.group(1), to='year', lang=lang), text)
    text = SUBTITLE_TEXT_TO_PLAIN_NUMBERS.sub(lambda n: _num2words_cardinal(n.group(1), lang), text)
    text = SUBTITLE_TEXT_TO_PLAIN_ABBREV_PATTERN.sub(lambda m: SUBTITLE_TEXT_TO_PLAIN_ABBREV_DICT[m.group(1).lower()],
                                                     text)

    text = SUBTITLE_TEXT_TO_PLAIN_REMOVE.sub("", text)
    text = SUBTITLE_TEXT_TO_PLAIN_WS.sub(" ", text)
    text = SUBTITLE_TEXT_TO_PLAIN_SQUEEZE_WS.sub(" ", text)
    text = text.lower().strip()
    return text


def _is_transcribed_word_suspicious(event: SubRipItem, lang: str = 'eng') -> bool:
    return _is_transcribed_word_ambiguous(event.text, lang) and event.duration.ordinal > 700


def _is_transcribed_word_ambiguous(word: str, lang: str = 'eng') -> bool:
    return word in ['the', 'in', 'is', 'an', 'yeah', 'that', 'hm', 'hmm', 'to']


def srt_words_to_sentences(words: list[SubRipItem]) -> list[SubRipItem]:
    """
    Collects words into "sentences". There may be multiple sentences per language rules, but we're calling
    sentences a collection of words.
    # TODO: Add capitalization and punctuation. Use hunspell to identify possible proper names.
    """
    chars_per_sentence = 40
    linebreaks_per_sentence = 2
    newline = '\n'
    words = list(filter(lambda e: not _is_transcribed_word_suspicious(e), words))
    result = []
    s = None
    for word in words:
        if s is not None:
            if word.start.ordinal - s.end.ordinal > SILENCE_FOR_NEW_SENTENCE:
                s = None
            else:
                split = (s.text + ' ' + word.text).split(newline)
                if len(split[-1]) > chars_per_sentence:
                    if len(split) >= linebreaks_per_sentence:
                        s = None
                    else:
                        s.text = s.text + newline

        if s is None:
            s = SubRipItem(index=len(result), start=word.start, end=word.end, text=word.text)
            result.append(s)
        else:
            s.end = word.end
            s.text = s.text + ' ' + word.text

    return result


def fix_subtitle_audio_alignment(subtitle_inout: Union[AssFile, SubRipFile], words: SubRipFile, lang='en',
                                 should_add_new_events=False) -> bool:
    """
    Fix the subtitle to be aligned to the audio using the transcribed words.
    :param subtitle_inout: the subtitle to align, srt or ssa
    :param words: the transcribed words, srt
    :param lang: language
    :param should_add_new_events: True to add new events from unmatched text
    :return: True if changes were made
    """

    # an offset is defined as the number of seconds between the input subtitle and the matched sentence in 'words'

    # the max number of seconds the 'words' are searched for a matching sentence
    max_offset_ms = 5000
    # minimum fuzz ratios to consider a sentence to match
    min_fuzz_ratios = [88, 85, 80, 70]
    # percentage of words to fuzz +/- from each input subtitle element
    word_count_fuzz_pct = 0.40
    unclaimed_word_capture_duration_max_ms = 1800
    # multiple runs work, it would be better to fix so a single pass works *shrug
    passes = 2

    subtitle_facade = subtitle.new_subtitle_file_facade(subtitle_inout)

    words_filtered = list(filter(lambda e: not _is_transcribed_word_suspicious(e), words))
    logger.debug("Removed %i suspicious words from transcription", len(words) - len(words_filtered))
    words_start_ms = list(map(lambda e: e.start.ordinal, words_filtered))
    words_start_end_ms = list(map(lambda e: (e.start.ordinal, e.end.ordinal), words_filtered))

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

    def get_adjustment_stats():
        adjustments = []
        for event_idx, event in enumerate(events):
            adjustments.append((abs(original_range_ms[event_idx][0] - event.start()),
                                abs(original_range_ms[event_idx][1] - event.end())))
        max_start_adjustment = max(map(lambda e: e[0], adjustments))
        ave_start_adjustment = average(list(map(lambda e: e[0], adjustments)))
        max_end_adjustment = max(map(lambda e: e[1], adjustments))
        ave_end_adjustment = average(list(map(lambda e: e[1], adjustments)))
        return adjustments, max_start_adjustment, ave_start_adjustment, max_end_adjustment, ave_end_adjustment

    def suspicious_last_word(event_idx: int, start_word_idx: int, end_word_idx: int) -> bool:
        if end_word_idx <= start_word_idx:
            return False
        start_ms = words_filtered[start_word_idx].start.ordinal
        last_word = words_filtered[end_word_idx - 1]
        if last_word.end.ordinal - start_ms <= original_duration_ms[event_idx]:
            return False
        if events[event_idx].normalized_text().endswith(last_word.text):
            return False
        return _is_transcribed_word_ambiguous(last_word.text, lang)

    def suspicious_first_word(event_idx: int, start_word_idx: int, end_word_idx: int) -> bool:
        if end_word_idx <= start_word_idx:
            return False
        first_word = words_filtered[start_word_idx]
        end_ms = words_filtered[end_word_idx - 1].end.ordinal
        if end_ms - first_word.start.ordinal <= original_duration_ms[event_idx]:
            return False
        if events[event_idx].normalized_text().startswith(first_word.text):
            return False
        return _is_transcribed_word_ambiguous(first_word.text, lang)

    def mark_claimed_words(event: subtitle.SubtitleElementFacade, unclaimed_word_events: list):
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
    inserted_event = []
    for event_idx, event in subtitle_facade.events():
        event.set_normalized_text(_subtitle_text_to_plain(event.text()))
        events.append(event)
        original_range_ms.append((event.start(), event.end()))
        original_duration_ms.append(event.duration())
        inserted_event.append(False)
    new_events_count = 0
    changed = False

    if len(events) == 0 or len(words_filtered) == 0:
        return False

    align_progress = progress.progress("subtitle alignment", 0, (len(min_fuzz_ratios) + 4) * passes)

    # TODO: original event that is too quiet to be picked up by transcribing
    # TODO: omitted event that transcribing picked up that should be combined with existing event
    last_fuzz_ratio = min_fuzz_ratios[-1]
    for pass_num in range(1, passes + 1):
        changed = False
        found_range_ms: list[Union[None, Tuple[int, int]]] = [None] * len(events)
        words_claimed = [False for _ in range(len(words_filtered))]
        progress_base = (pass_num - 1) * (len(min_fuzz_ratios) + 4)

        # check if current ranges are good fits
        for event_idx, event in enumerate(events):
            if found_range_ms[event_idx] is not None:
                continue
            try:
                current_start_idx, current_end_idx, current_text = get_transcription_info(event)
                event_text = event.normalized_text()
                if len(event_text) > 3 and fuzz.ratio(event_text, current_text) >= 96:
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

        # TODO: need to keep sound effects, perhaps match empty space in transcription to similar length of effect
        # single words match more things erroneously than longer strings of words
        for word_count_min in [8, 5, 3, 2, 1]:
            match_attempt = [False for _ in range(0, len(events))]
            # run through each ratio and iteratively find matches, using existing matches to narrow the search
            for min_fuzz_ratio_idx, min_fuzz_ratio in enumerate(min_fuzz_ratios):
                for event_idx, event in enumerate(events):
                    if match_attempt[event_idx] or found_range_ms[event_idx] is not None:
                        continue

                    # TODO: match permutations of text for differing numbers (digits vs. proper spoken), abbreviations, etc.
                    # TODO: permutations for slang short hand, i.e. "outta" for "out of"
                    # TODO: permutations for ("know", "no"), etc.
                    event_text = event.normalized_text()
                    word_count = len(event_text.split())
                    if word_count < word_count_min:
                        continue

                    match_attempt[event_idx] = True
                    word_counts = range(floor(word_count * (1.0 - word_count_fuzz_pct)),
                                        ceil(word_count * (1.0 + word_count_fuzz_pct)) + 1)
                    logger.debug("Matching sentence from event %i (%s,%s) '%s', (%i,%i) words",
                                 event_idx, common.ms_to_ts(event.start()), common.ms_to_ts(event.end()), event_text,
                                 word_counts.start, word_counts.stop)
                    # ignore ranges that have already been matched
                    start_search_ms = max(event.start() - max_offset_ms,
                                          max(map(lambda e: e[1],
                                                  filter(lambda f: f is not None, found_range_ms[:event_idx])),
                                              default=0))
                    end_search_ms = min(event.start() + max_offset_ms,
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
                    logger.debug("candidates are %s", candidates)
                    if not candidates:
                        logger.debug("no candidates for min_fuzz_ratio %i '%s' (%i,%i)",
                                     min_fuzz_ratio, event_text, start_search_ms, end_search_ms)
                        continue
                    else:
                        matches = fuzzprocess.extractBests(event_text, candidates, scorer=fuzz.ratio,
                                                           score_cutoff=last_fuzz_ratio)

                        # remove overlapping matches with lesser ratio
                        if matches:
                            matches.sort(reverse=True, key=lambda e: [e[1], -1*e[2][0]])
                            logger.debug("event %i matches before removing overlaps: %s", event_idx, matches)
                            for matches_idx, match in enumerate(matches.copy()):
                                matches_idx2 = matches_idx + 1
                                while matches_idx2 < len(matches):
                                    m = matches[matches_idx2]
                                    if match[2][0] <= m[2][1] and m[2][0] <= match[2][1]:
                                        # overlap
                                        matches.pop(matches_idx2)
                                    else:
                                        matches_idx2 += 1
                            logger.debug("event %i matches after removing overlaps: %s", event_idx, matches)

                        # check for similar sentences in which the later matches better with the former transcription
                        last_fuzz_ratio_count = len(matches)
                        if last_fuzz_ratio_count > 1 and min_fuzz_ratio != last_fuzz_ratio:
                            matches_min_fuzz_ratio = min(filter(lambda e: e[1], matches))[1]
                            if matches_min_fuzz_ratio < min_fuzz_ratio:
                                logger.debug("event %i multiple transcriptions may match %s, skipping", event_idx,
                                             list(map(lambda e: e[1], matches)))
                                continue

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
                        changed = True
                    else:
                        logger.debug("not matched r%i for '%s', candidates are %s",
                                     min_fuzz_ratio, event_text, str(candidates)[:200])
            align_progress.progress(progress_base + min_fuzz_ratio_idx + 1)

        logger.debug("matched count %i/%i", len(list(filter(lambda e: e is not None, found_range_ms))),
                     len(found_range_ms))

        # we've matched all we can, now do something with unclaimed words
        unclaimed_word_events = []
        for word_idx, word in enumerate(words_filtered):
            if not words_claimed[word_idx]:
                unclaimed_word_events.append((word_idx, word))
        adjustments, max_start_adjustment, ave_start_adjustment, max_end_adjustment, ave_end_adjustment = get_adjustment_stats()

        # 1. attach unclaimed words to matched events that are missing duration
        # 2. adjust duration for matched events that have sound effects
        for event_idx, event in enumerate(events):
            if found_range_ms[event_idx] is None:
                continue
            missing_duration = original_duration_ms[event_idx] - event.duration()
            if missing_duration <= 0:
                continue
            if event.is_sound_effect():
                continue
            try:
                start_word_idx, end_word_idx, transcribed_text = get_transcription_info(event)
            except ValueError:
                continue
            word_count = len(event.normalized_text().split())
            transcribed_word_count = len(transcribed_text.split())

            event_moved = True
            while event_moved:
                event_moved = False

                new_start_word_idx = None
                new_start_ordinal = None
                beginning_duration = None
                # capture possible missing words
                if (transcribed_word_count < word_count
                        and start_word_idx > 0
                        and not words_claimed[start_word_idx-1]):
                    beginning_duration = words_filtered[start_word_idx].start.ordinal - words_filtered[start_word_idx-1].start.ordinal
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
                        beginning_duration = words_filtered[start_word_idx].start.ordinal - words_filtered[start_word_idx-1].end.ordinal
                        if beginning_duration <= missing_duration:
                            new_start_ordinal = words_filtered[start_word_idx-1].end.ordinal + (missing_duration - beginning_duration)

                new_end_word_idx = None
                new_end_ordinal = None
                ending_duration = None
                # capture possible missing words
                if (transcribed_word_count < word_count
                        and end_word_idx < len(words_claimed) - 1
                        and not words_claimed[end_word_idx+1]):
                    ending_duration = words_filtered[end_word_idx+1].end.ordinal - words_filtered[end_word_idx].end.ordinal
                    if ending_duration <= missing_duration:
                        new_end_word_idx = end_word_idx - 1
                        new_end_ordinal = words_filtered[end_word_idx].end.ordinal
                # capture empty space for sound effects
                if new_end_word_idx is None and event.has_ending_sound_effect():
                    if end_word_idx < len(words_claimed) - 1:
                        ending_duration = words_filtered[end_word_idx+1].start.ordinal - words_filtered[end_word_idx].end.ordinal
                        if ending_duration <= missing_duration:
                            new_end_ordinal = words_filtered[end_word_idx+1].start.ordinal - (missing_duration - ending_duration)

                # capture the largest missing duration at either beginning or end
                if new_start_ordinal is not None and ((new_end_ordinal is not None and beginning_duration >= ending_duration) or new_end_ordinal is None):
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
                elif new_end_ordinal is not None and ((new_start_ordinal is not None and beginning_duration < ending_duration) or new_start_ordinal is None):
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

        # 3. events with only sound effects, find gap roughly matching duration
        # 4. move unmatched events based on unclaimed words
        # 5. with no transcription help, move event based on average adjustment

        for event_idx, event in enumerate(events):
            try:
                if event_idx == 0:
                    continue
                matched = found_range_ms[event_idx] is not None
                previous_matched = found_range_ms[event_idx - 1] is not None if event_idx > 0 else False
                event_previous = events[event_idx - 1]
                if not matched:
                    event_moved = False
                    if len(event.normalized_text()) > 0:
                        # try to start non-matches at the beginning of an unclaimed word
                        # TODO: unclaimed word could be part of previous match
                        unclaimed_word = next(filter(
                            lambda e: e[1].start.ordinal >= event_previous.end(), unclaimed_word_events), None)
                        if unclaimed_word is not None and len(unclaimed_word[1].text) > 3:
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
                        if abs(event_previous_adj) < ave_start_adjustment:
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
                        event.move(event_previous.end())
                    elif event_idx > 1:
                        event_previous.set_start(events[event_idx - 2].end())
                if event_previous.end() > event.start():
                    logger.debug("event %i end overlaps %i start, %s > %s, matched? %r, previous matched? %r",
                                 event_idx - 1, event_idx,
                                 common.ms_to_ts(event_previous.end()), common.ms_to_ts(event.start()),
                                 matched, previous_matched)
                    if previous_matched:
                        event.move(event_previous.end())
                    else:
                        event_previous.set_end(event.start())

                mark_claimed_words(event, unclaimed_word_events)
            finally:
                event_previous = event

        align_progress.progress(progress_base + len(min_fuzz_ratios) + 1)

        # collect large missing events, but insert later so we don't have to fix up arrays containing info
        new_events = []
        last_word_idx = -1
        for event_idx, event in enumerate(events):
            try:
                start_word_idx, end_word_idx, transcribed_text = get_transcription_info(event)
            except ValueError:
                continue

            if 0 <= last_word_idx < (start_word_idx - 1):
                unclaimed_words = []
                for i in range(last_word_idx, start_word_idx):
                    if not words_claimed[i]:
                        unclaimed_words.append(words_filtered[i])
                if len(unclaimed_words) > 0:
                    new_events.insert(0, (event_idx, unclaimed_words))

            last_word_idx = end_word_idx

        align_progress.progress(progress_base + len(min_fuzz_ratios) + 2)

        if should_add_new_events and len(new_events) > 0:
            for new_event in new_events:
                insert_idx = new_event[0]
                unclaimed_words = new_event[1]

                for sentence in srt_words_to_sentences(unclaimed_words):
                    event = subtitle_facade.insert(insert_idx)
                    event.set_start(sentence.start.ordinal)
                    event.set_end(sentence.end.ordinal)
                    event.set_text(sentence.text)
                    event.set_normalized_text(_subtitle_text_to_plain(sentence.text))
                    events.insert(insert_idx, event)
                    logger.debug("inserted new event at %i (%s,%s) '%s'", insert_idx,
                                 common.ms_to_ts(event.start()), common.ms_to_ts(event.end()), event.log_text())
                    new_events_count += 1
                    found_range_ms.insert(insert_idx, (sentence.start.ordinal, sentence.end.ordinal))
                    original_range_ms.insert(insert_idx, (sentence.start.ordinal, sentence.end.ordinal))
                    original_duration_ms.insert(insert_idx, sentence.end.ordinal - sentence.start.ordinal)
                    inserted_event.insert(insert_idx, True)
                    insert_idx += 1

            # ensure monotonically increasing indicies
            for event_idx, event in enumerate(events):
                event.set_index(event_idx + 1)

        align_progress.progress(progress_base + len(min_fuzz_ratios) + 3)

        # extend durations based on original, not to overlap
        # TODO: consider sound effects in subtitles
        for event_idx, event in enumerate(events[:-1]):
            if inserted_event[event_idx]:
                continue
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

        align_progress.progress(progress_base + len(min_fuzz_ratios) + 4)

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
            ratio = fuzz.ratio(event.normalized_text(), transcribed_text)
            matches = "matches" if found_range_ms[event_idx] is not None else "claims"
            logger.log(logging.WARNING if ratio < 50 else logging.DEBUG,
                       "event %i (%s%+i - %s%+i %i ms was %i ms) '%s' %s words '%s' with ratio %i",
                       event_idx,
                       common.ms_to_ts(event.start()), event.start() - original_range_ms[event_idx][0],
                       common.ms_to_ts(event.end()), event.end() - original_range_ms[event_idx][1],
                       event.duration(), original_duration_ms[event_idx],
                       event.log_text(), matches, transcribed_text, ratio)

    # check stats for material changes
    # use these stats to determine if we adjusted enough to make a difference and return False if not
    adjustments, max_start_adjustment, ave_start_adjustment, max_end_adjustment, ave_end_adjustment = get_adjustment_stats()
    logger.info(
        "subtitle alignment stats: max_start_adjustment %i, max_end_adjustment %i, ave_start_adjustment %i, ave_end_adjustment %i, new events %i",
        max_start_adjustment, max_end_adjustment, ave_start_adjustment, ave_end_adjustment, new_events_count)
    for event_idx, adjustment in enumerate(adjustments):
        adjustment_log_level = 0
        if adjustment[0] > max_start_adjustment / 2 or adjustment[1] > max_end_adjustment / 2:
            adjustment_log_level = logging.WARNING
        if events[event_idx].start() >= events[event_idx].end():
            adjustment_log_level = logging.ERROR
        if adjustment_log_level > 0:
            logger.log(adjustment_log_level,
                       "subtitle alignment stats: event %i adjustment (%i,%i) (%s - %s) '%s'",
                       event_idx,
                       events[event_idx].start() - original_range_ms[event_idx][0],
                       events[event_idx].end() - original_range_ms[event_idx][1],
                       common.ms_to_ts(events[event_idx].start()), common.ms_to_ts(events[event_idx].end()),
                       events[event_idx].log_text())
    if ave_start_adjustment < 100 and ave_end_adjustment < 100:
        changed = False

    align_progress.stop()

    return changed


def profanity_filter_cli(argv):
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
             "mute-voice-channels", "mute-all-channels", "verbose"])
    except getopt.GetoptError:
        usage()
        sys.exit(CMD_RESULT_ERROR)
    for opt, arg in opts:
        if opt == '--help':
            usage()
            sys.exit(CMD_RESULT_ERROR)
        elif opt in ("-n", "--dry-run"):
            dry_run = True
        elif opt in ("-k", "--keep"):
            keep = True
        elif opt in ("-d", "--debug"):
            debug = True
        elif opt == "--verbose":
            logging.getLogger().setLevel(logging.DEBUG)
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
        sys.exit(CMD_RESULT_ERROR)

    input_file = args[0]

    atexit.register(common.finish)

    sys.exit(
        profanity_filter(input_file, dry_run=dry_run, keep=keep, force=force, filter_skip=filter_skip,
                         mark_skip=mark_skip, unmark_skip=unmark_skip, workdir=workdir, verbose=True,
                         mute_channels=mute_channels, language=language))


if __name__ == '__main__':
    common.cli_wrapper(profanity_filter_cli)
