#!/usr/bin/env python3

import atexit
import copy
import getopt
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pysrt
from ass_parser import read_ass, write_ass, AssEventList, CorruptAssLineError
from numpy import loadtxt
from pysrt import SubRipItem, SubRipFile, SubRipTime

import common
from common import subtitle, tools, config, constants

# Increment when a coding change materially effects the output
FILTER_VERSION = 11
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
--keep
    Keep original file in a backup prefixed by ".~"
--debug
    Only create hidden output file for debugging
--work-dir={config.get_work_dir()}
--remove
    Remove filtering
--mark-skip
    Mark file(s) for the filter to skip. If a file has been filtered, filtering will be removed. 
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
                        unmark_skip=None, language=constants.LANGUAGE_ENGLISH, workdir=None, verbose=False) -> int:
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
    common.TEMPFILENAME = os.path.join(dir_filename,
                                       f".~{'.'.join(base_filename.split('.')[0:-1])}.transcoded.{input_type}")
    # Hides filename from user UI and Dropbox
    common.HIDDEN_FILENAME = f"{dir_filename}/.~{base_filename}"
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

    streams_file = 0

    #
    # Construct command
    #

    # Global arguments
    arguments = []
    if not verbose:
        arguments.extend(["-loglevel", "error"])
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
    audio_to_text_filter = None

    if need_original_subtitle_ocr(subtitle_original=subtitle_original,
                                  media_duration=float(input_info[constants.K_FORMAT][constants.K_DURATION]),
                                  force=force):
        subtitle_srt_generated = ocr_subtitle_bitmap_to_srt(input_info, temp_base, language, verbose=verbose)

    if audio_original and need_original_subtitle_transcribed(subtitle_original=subtitle_original,
                                                             subtitle_words=subtitle_words,
                                                             current_audio2text_version=current_audio2text_version,
                                                             media_duration=float(
                                                                 input_info[constants.K_FORMAT][constants.K_DURATION]),
                                                             force=force):
        # We may only need the words from transcription
        if subtitle_srt_generated or subtitle_original:
            logger.info("Transcribing for words")
        else:
            logger.info("Transcribing for text and words")
        audio_channels = int(audio_original.get(constants.K_CHANNELS, 0))
        if audio_channels > 2:
            audio_to_text_filter = 'pan=1c|FC<0.3*FL+FC+0.3*FR,anlmdn'
        else:
            audio_to_text_filter = 'anlmdn'
        _srt_text, subtitle_srt_words = audio_to_srt(input_info, audio_original, workdir, audio_to_text_filter,
                                                     language, verbose=verbose)
        if _srt_text and os.stat(_srt_text).st_size > 0:
            audio_to_text_version = AUDIO_TO_TEXT_VERSION
            if not subtitle_original and subtitle_srt_generated is None:
                subtitle_srt_generated = _srt_text

    tags = input_info[constants.K_FORMAT].get(constants.K_TAGS, {}).copy()
    tags[constants.K_FILTER_HASH] = filter_hash
    tags[constants.K_FILTER_VERSION] = FILTER_VERSION
    tags[constants.K_AUDIO_TO_TEXT_VERSION] = audio_to_text_version if audio_to_text_version else ''
    tags[constants.K_AUDIO_TO_TEXT_FILTER] = audio_to_text_filter if audio_to_text_filter else ''
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
    subtitle_words_filename = f"{temp_base}.words.{subtitle_codec}"
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
        if verbose:
            logger.info(f"Removing filtered streams")
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
        arguments.extend(["-c:s", "copy"])

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
                              "-disposition:s:0", "default"])
            subtitle_output_idx = 1
        else:
            subtitle_output_idx = 0
        if subtitle_srt_words is not None:
            arguments.extend(["-i", subtitle_srt_words,
                              "-map", f"{streams_file + 2}:0",
                              f"-metadata:s:s:1", f'title={constants.TITLE_WORDS}',
                              f"-disposition:s:1", "-default+metadata"])
            subtitle_output_idx += 1
        elif subtitle_words_idx is not None:
            arguments.extend(["-map", f"{streams_file}:{subtitle_words_idx}",
                              f"-metadata:s:s:{subtitle_output_idx}", f'title={constants.TITLE_WORDS}',
                              f"-disposition:s:{subtitle_output_idx}", "-default+metadata"])
            subtitle_output_idx += 1
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
            ass_data_forced = copy.copy(ass_data)
            ass_data_forced.events = AssEventList()
            for event in list(ass_data.events):
                original_text = event.text
                filtered_text, stopped = filter_text(censor_list, stop_list, allow_list, original_text)
                if filtered_text != original_text:
                    event.text = filtered_text
                    ass_data_forced.events.append(copy.copy(event))
                    filtered_spans.append([event.start, event.end])
                    if stopped:
                        stopped_spans.append([event.start, event.end])
            write_ass(ass_data, Path(subtitle_filtered_filename))
            write_ass(ass_data_forced, Path(subtitle_filtered_forced_filename))
        elif subtitle_codec in [constants.CODEC_SUBTITLE_SRT, constants.CODEC_SUBTITLE_SUBRIP]:
            srt_data = pysrt.open(subtitle_original_filename)
            srt_data_forced = copy.copy(srt_data)
            srt_data_forced.data = []
            for event in list(srt_data):
                original_text = event.text
                filtered_text, stopped = filter_text(censor_list, stop_list, allow_list, original_text)
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
            logger.info(f"No changes, updating filter hash")
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
        # TODO: For surround sound, filter only the channel in which we expect speech
        # https://stackoverflow.com/questions/33533401/volume-adjust-and-channel-merge-on-video-using-ffmpeg
        if len(filtered_spans) > 0:
            mute_filters = []
            for span in filtered_spans:
                if isinstance(span[0], int):
                    mute_filters.append(f"volume=enable='between(t,{span[0] / 1000.0},{span[1] / 1000.0})':volume=0")
                else:
                    mute_filters.append(f"volume=enable='between(t,{span[0] / 1000.0},{span[1] / 1000.0})':volume=0")
            arguments.extend(["-map", f"{streams_file}:{audio_original_idx}",
                              f"-c:a:{audio_output_idx}", "libopus",
                              f"-metadata:s:a:{audio_output_idx}", f'title={constants.TITLE_FILTERED}',
                              f"-disposition:a:{audio_output_idx}", "default"])
            common.extend_opus_arguments(arguments, input_info['streams'][audio_original_idx], f'a:{audio_output_idx}',
                                         mute_filters)
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
            arguments.extend(["-map", f"{subtitle_srt_generated_file}:0"])
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
                              f"-metadata:s:s:{subtitle_output_idx}", f'language={language}',
                              f"-disposition:s:{subtitle_output_idx}", "-default+metadata"])
            subtitle_output_idx += 1
        elif subtitle_words_idx is not None:
            arguments.extend(["-map", f"{streams_file}:{subtitle_words_idx}"])
            arguments.extend([f"-metadata:s:s:{subtitle_output_idx}", f'title={constants.TITLE_WORDS}',
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

    arguments.append(common.TEMPFILENAME)

    if dry_run:
        logger.info(tools.ffmpeg.array_as_command(arguments))
        return CMD_RESULT_FILTERED

    # Check again because there is time between the initial check and when we write to the file
    if common.assert_not_transcoding(input_file, exit=False) != 0:
        return CMD_RESULT_ERROR
    try:
        Path(common.TEMPFILENAME).touch(mode=0o664, exist_ok=False)
    except FileExistsError:
        return CMD_RESULT_ERROR

    logger.info(f"Starting filtering of {filename} to {common.TEMPFILENAME}")
    logger.info(tools.ffmpeg.array_as_command(arguments))
    tools.ffmpeg.run(arguments, check=True)

    if os.stat(common.TEMPFILENAME).st_size == 0:
        logger.fatal(f"Output at {common.TEMPFILENAME} is zero length")
        return CMD_RESULT_ERROR

    #
    # Encode Done. Performing Cleanup
    #
    logger.info(f"Finished filtering of {filename} to {common.TEMPFILENAME}")

    common.match_owner_and_perm(target_path=common.TEMPFILENAME, source_path=filename)

    # Hide original file in case OUTPUT_TYPE is the same as input
    if not debug:
        os.replace(filename, common.HIDDEN_FILENAME)
    try:
        os.replace(common.TEMPFILENAME, output_filename)
    except OSError:
        # Put original file back as fall back
        os.replace(common.HIDDEN_FILENAME, filename)
        logger.fatal(f"Failed to move converted file: {common.TEMPFILENAME}")
        return CMD_RESULT_ERROR

    if not keep and not debug:
        os.remove(common.HIDDEN_FILENAME)

    logger.info("Filtering done")

    return CMD_RESULT_FILTERED


def load_censor_list():
    # TODO: allow multiple lists based on 'levels' (R, PG, G ?) and allow selection at runtime
    result = loadtxt(os.path.join(os.path.dirname(common.__file__), 'censor_list.txt'), dtype='str', delimiter='\xFF')
    result = list(filter(phrase_list_accept_condition, result))
    result.sort(key=lambda e: len(re.sub('[^A-Za-z]+', '', e)), reverse=True)
    result = list(map(lambda e: phrase_to_pattern(e), result))
    return result


def load_stop_list():
    # TODO: allow multiple lists based on 'levels' (R, PG, G ?) and allow selection at runtime
    result = loadtxt(os.path.join(os.path.dirname(common.__file__), 'stop_list.txt'), dtype='str', delimiter='\xFF')
    result = list(filter(phrase_list_accept_condition, result))
    result = list(map(lambda e: phrase_to_pattern(e), result))
    return result


def load_allow_list():
    result = loadtxt(os.path.join(os.path.dirname(common.__file__), 'allow_list.txt'), dtype='str', delimiter='\xFF')
    result = list(filter(phrase_list_accept_condition, result))
    result = list(map(lambda e: phrase_to_pattern(e), result))
    return result


def compute_filter_hash(censor_list=None, stop_list=None, allow_list=None):
    if censor_list is None:
        censor_list = load_censor_list()
    if stop_list is None:
        stop_list = load_stop_list()
    if allow_list is None:
        allow_list = load_allow_list()
    hash_input = str(FILTER_VERSION) + ','.join(censor_list) + ','.join(stop_list) + ','.join(allow_list)
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
            duration = common.parse_edl_ts(duration_s)
            if duration < (media_duration * 0.60):
                return True
    except:
        pass

    return force


def need_original_subtitle_transcribed(subtitle_original: dict, subtitle_words: dict, current_audio2text_version: str,
                                       media_duration: float, force: bool) -> bool:
    """
    Determine if the original subtitle needs transcribed from audio.
    :param subtitle_original:
    :param current_audio2text_version:
    :return: True to transcribe
    """
    if not subtitle_original:
        return True
    if not subtitle_words:
        return True

    duration_s = subtitle_original.get('tags', {}).get('DURATION', '')
    try:
        if duration_s:
            duration = common.parse_edl_ts(duration_s)
            if duration < (media_duration * 0.60):
                return True
    except:
        pass

    if force and current_audio2text_version not in [None, '']:
        return True
    if current_audio2text_version not in [None, '', str(AUDIO_TO_TEXT_VERSION)]:
        return True

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
        extract_command = ["-loglevel", "error", '-hide_banner', '-y', '-analyzeduration',
                           common.ANALYZE_DURATION,
                           '-probesize', common.PROBE_SIZE,
                           '-i', input_info['format']['filename'],
                           '-map', f'0:{bluray[constants.K_STREAM_INDEX]}', '-c:s', 'copy',
                           subtitle_filename
                           ]
        if verbose:
            logger.info(tools.ffmpeg.array_as_command(extract_command))
        tools.ffmpeg.run(extract_command, check=True, capture_output=True)

    dvd = find_subtitle_dvdsub(input_info, language)
    if dvd:
        dvd = dvd[0]
        subtitle_filename = f"{temp_base}.vob"
        if not debug:
            common.TEMPFILENAMES.append(subtitle_filename)
        extract_command = ["-loglevel", "error", '-hide_banner', '-y', '-analyzeduration',
                           common.ANALYZE_DURATION,
                           '-probesize', common.PROBE_SIZE,
                           '-i', input_info['format']['filename'],
                           '-map', f'0:{dvd[constants.K_STREAM_INDEX]}', '-c:s', 'dvdsub',
                           subtitle_filename
                           ]
        if verbose:
            logger.info(tools.ffmpeg.array_as_command(extract_command))
        tools.ffmpeg.run(extract_command, check=True, capture_output=True)

    if subtitle_filename is None:
        return None

    tools.subtitle_edit.run([subtitle_filename], check=True)
    if not os.access(subtitle_srt_filename, os.R_OK):
        logger.error(f"SRT not generated from OCR")
        return None

    word_found_pct = words_in_dictionary_pct(subtitle_srt_filename, language)
    if word_found_pct < WORD_FOUND_PCT_THRESHOLD:
        logger.error(f"OCR text appears to be incorrect, {word_found_pct}% words found in {language} dictionary")
        return None

    # TODO: Add "OCR by media-hare+SubtitleEdit" at beginning and end

    return subtitle_srt_filename


def audio_to_srt(input_info: dict, audio_original: dict, workdir, audio_filter: str = None, language=None,
                 verbose=False) -> (str, str):
    """
    Attempts to create a text subtitle from the original audio stream.
    1. vosk does not seem to like filenames with spaces, it's thrown a division by zero
    2. audio stream is being converted to AC3 stereo with default ffmpeg bitrate (192kbps) for most compatibility
    3. --tasks is being set but so far it doesn't seem to yield more cores used
    :return: srt filename for subtitle, srt filename for words) or None
    """
    global debug

    try:
        from vosk import Model, KaldiRecognizer, GpuInit, GpuThreadInit
        GpuInit()
        GpuThreadInit()
    except ImportError as e:
        logger.warning("Cannot transcribe audio, vosk missing")
        return None

    freq = 16000
    chunk_size = 4000
    words_per_line = 7

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
    audio_process = tools.ffmpeg.Popen(extract_command, stdout=subprocess.PIPE)

    _vosk_language = vosk_language(language)
    model = Model(model_name=vosk_model(_vosk_language), lang=_vosk_language)
    rec = KaldiRecognizer(model, freq)
    rec.SetWords(True)

    audio_process.stdout.read(44)  # skip header
    results = []
    while True:
        data = audio_process.stdout.read(chunk_size)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if 'result' in result:
                results.append(result)
                if 'text' in result:
                    logger.debug("text: %s", result['text'])
    result = json.loads(rec.FinalResult())
    if 'result' in result:
        results.append(result)
        if 'text' in result:
            logger.debug("text: %s", result['text'])
    audio_process.stdout.close()
    extract_return_code = audio_process.wait()
    if extract_return_code != 0:
        logger.error("Cannot transcribe audio, ffmpeg returned %s", extract_return_code)
        return None

    subs_words = []
    subs_text = []

    for i, res in enumerate(results):
        # 'the' by itself seems to be an artifact of soundtrack / background noise
        if res.get('text') == 'the':
            continue
        words = res['result']
        for word in words:
            s = SubRipItem(index=len(subs_words),
                           start=SubRipTime(seconds=word['start']),
                           end=SubRipTime(seconds=word['end']),
                           text=word['word'])
            subs_words.append(s)

        # TODO: break by character count?
        for j in range(0, len(words), words_per_line):
            line = words[j: j + words_per_line]
            s = SubRipItem(index=len(subs_text),
                           start=SubRipTime(seconds=line[0]['start']),
                           end=SubRipTime(seconds=line[-1]['end']),
                           text=' '.join([l['word'] for l in line]))
            subs_text.append(s)

    srt_words = SubRipFile(items=subs_words, path=words_filename)
    srt_words.save(Path(words_filename), 'utf-8')

    srt = SubRipFile(items=subs_text, path=subtitle_srt_filename)
    audio_to_text_cleanup(srt)
    srt.save(Path(subtitle_srt_filename), 'utf-8')

    word_found_pct = words_in_dictionary_pct(subtitle_srt_filename, language)
    if word_found_pct < WORD_FOUND_PCT_THRESHOLD:
        logger.error(
            f"audio-to-text transcription appears to be incorrect, {word_found_pct}% words found in {language} dictionary")
        return None

    # TODO: Add "transcribed by media-hare+Vosk" at beginning and end

    return subtitle_srt_filename, words_filename


def audio_to_text_cleanup(srt_data: SubRipFile) -> None:
    """
    Post process audio to text subtitles. This cleanup is tuned closely to the vosk models and may need to be changed
    when updated models are used.
      - 'the' by itself
      - ' the' at the end
    """
    for i in range(len(srt_data) - 1, -1, -1):
        event = srt_data[i]
        cleaned_text = event.text
        while cleaned_text.lower().endswith(' the'):
            cleaned_text = cleaned_text[0:-4]
        if cleaned_text.lower() == 'the':
            cleaned_text = ''
        if len(cleaned_text) == 0:
            logger.info("Removed empty SRT event: %s", str(event.start))
            del srt_data[i]
        elif cleaned_text != event.text:
            logger.info("Cleaned SRT event: %s, %s -> %s", str(event.start), event.text, cleaned_text)
            event.text = cleaned_text


def words_in_dictionary_pct(subtitle_srt_filename, language):
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
            for word in re.sub('[^A-Za-z\' ]+', ' ', event.text).split():
                word_count += 1
                if hobj.spell(word):
                    word_found_count += 1
        if word_count < 100:
            logger.warning("word count less than 100, returning 0%")
            return 0.0
        word_found_pct = 100.0 * float(word_found_count) / float(word_count)
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


STOP_CLEAN_PATTERN = "^((?:[{].*?[}])*)"
# Identifies a word break in both SRT and ASS. ASS is more complex than SRT and has it's own markup.
WORD_BREAKS = r'((?:\\s|[,!]|[{].+?[}]|[\\\\]x[0-9a-fA-F]+|[\\\\][Nh])+|\\b|$)'
# When the censor list uses the regex "^" for the beginning of text, this is substituted to make it work
NO_PREVIOUS_WORD = r'(?<!\\w\\s)'
# Matches subtitles that have text already filtered using various symbols
PRE_FILTERED = r'[!@#$%^&*+]\s?(?:[@#$%^&*+]\s?){2,}'
# The string to use for masking
MASK_STR = '***'


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


def matches_stop_pattern(stop_pattern, text, allow_ranges: list[tuple]) -> bool:
    for m in re.finditer(stop_pattern, text, flags=re.IGNORECASE):
        allowed = False
        for allow_range in allow_ranges:
            if allow_range[0] <= m.start(0) < allow_range[1] \
                    or allow_range[0] <= m.end(0) < allow_range[1]:
                allowed = True
                break
        if not allowed:
            return True
    return False


def filter_text(censor_list: list, stop_list: list, allow_list: list, text) -> (str, bool):
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
            for m in re.finditer(allow_pattern, text, flags=re.IGNORECASE):
                allow_ranges.append(m.span(0))
        except re.error as e:
            print(f"ERROR in allow list: {allow_pattern}")
            raise e

    if re.search(PRE_FILTERED, text, flags=re.IGNORECASE):
        text2 = re.sub(PRE_FILTERED, MASK_STR, text)
        if text2 == text:
            # We need the text to change to mute the audio
            text = text2 + ' _'
        else:
            text = text2
    for stop_pattern in stop_list:
        try:
            if matches_stop_pattern(stop_pattern, text, allow_ranges):
                text = re.search(STOP_CLEAN_PATTERN, text).expand(fr"\1{MASK_STR}")
                return text, True
        except re.error as e:
            print(f"ERROR in stop list: {stop_pattern}")
            raise e
    for contains_pattern in censor_list:
        try:
            text = re.sub(contains_pattern, lambda m: contains_pattern_repl(m, allow_ranges), text, flags=re.IGNORECASE)
        except re.error as e:
            print(f"ERROR in censor list: {contains_pattern}")
            raise e
    # clean up trailing punctuation
    text = re.sub(r'\*\*\*([,!.?\'’]+)', MASK_STR, text)
    # clean up redundant replacements
    text = re.sub(r'(\*\*\*[\s,]*)+\*\*\*', MASK_STR, text)
    return text, False


def phrase_to_pattern(phrase):
    """
    Transforms a phrase, which is roughly a regex, into something that will match both SRT and ASS markup.
    """
    phrase_fancy_word_breaks = re.sub(r'([^\s^$]+)\s*', r'(\1)' + WORD_BREAKS, phrase, flags=re.IGNORECASE)
    phrase_beginning_marker = re.sub(r'[\^]', NO_PREVIOUS_WORD, phrase_fancy_word_breaks)
    phrase_starting_word_break = re.sub(r'.', WORD_BREAKS, '.') + phrase_beginning_marker
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


def vosk_model(language: str) -> [None, str]:
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
    for key in [constants.K_FILTER_HASH, constants.K_FILTER_VERSION, constants.K_FILTER_STOPPED]:
        if key in tags:
            del tags[key]
    if common.should_replace_media_title(input_info):
        tags[constants.K_MEDIA_TITLE] = common.get_media_title_from_filename(input_info)
    tags[constants.K_MEDIA_PROCESSOR] = constants.V_MEDIA_PROCESSOR
    common.write_mkv_tags(tags, tags_filename)
    if not dry_run and not debug:
        tools.mkvpropedit.run([filename, "--tags", f"global:{tags_filename}"])
    return CMD_RESULT_MARKED


if __name__ == '__main__':
    common.setup_cli()
    argv = sys.argv[1:]

    dry_run = False
    keep = False
    force = False
    filter_skip = None
    mark_skip = None
    unmark_skip = None
    language = constants.LANGUAGE_ENGLISH
    workdir = config.get_work_dir()

    try:
        opts, args = getopt.getopt(
            list(argv), "nkdrf",
            ["dry-run", "keep", "debug", "remove", "mark-skip", "unmark-skip", "force", "work-dir="])
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

    if len(args) == 0:
        usage()
        sys.exit(CMD_RESULT_ERROR)

    input_file = args[0]

    atexit.register(common.finish)

    sys.exit(
        profanity_filter(input_file, dry_run=dry_run, keep=keep, force=force, filter_skip=filter_skip,
                         mark_skip=mark_skip, unmark_skip=unmark_skip, workdir=workdir, verbose=True))
