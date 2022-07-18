#!/usr/bin/env python3

import atexit
import copy
import getopt
import hashlib
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from shutil import which

import hunspell
import pysrt
from ass_parser import read_ass, write_ass, AssEventList
from numpy import loadtxt
from pysrt import SubRipItem

import common

# Increment when a coding change materially effects the output
FILTER_VERSION = 11

# exit code for content had filtering applied, file has been significantly changed
CMD_RESULT_FILTERED = 0
# exit code for content version mark has been updated, file has been trivially changed
CMD_RESULT_MARKED = 1
# exit code for content is unchanged
CMD_RESULT_UNCHANGED = 2
# exit code for general error
CMD_RESULT_ERROR = 255

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
- Set the tag 'PFILTER_SKIP' in the video to 'y' or 't' to skip filtering.
  ex: mkvpropedit filename.mkv --tags global:{os.path.join(os.path.join(os.path.dirname(os.path.dirname(common.__file__))), 'mkv-filter-skip.xml')}

--dry-run
    Output command that would be used but do nothing
--keep
    Keep original file in a backup prefixed by ".~"
--debug
    Only create hidden output file for debugging
--work-dir={common.get_work_dir()}
--remove
    Remove filtering
--force
    Force re-filtering
""", file=sys.stderr)


def profanity_filter(*args, **kwargs) -> int:
    try:
        return do_profanity_filter(*args, **kwargs)
    finally:
        common.finish()


def do_profanity_filter(input_file, dry_run=False, keep=False, force=False, filter_skip=None,
                        language=common.LANGUAGE_ENGLISH, workdir=None, verbose=False) -> int:
    mkvpropedit = common.find_mkvpropedit()
    if len(mkvpropedit) == 0:
        logger.fatal("'mkvpropedit' not found, install mkvtoolnix")
        return CMD_RESULT_ERROR

    if not os.path.isfile(input_file):
        logger.fatal(f"{input_file} does not exist")
        return CMD_RESULT_ERROR

    filename = os.path.realpath(input_file)
    base_filename = os.path.basename(filename)
    input_type = base_filename.split(".")[-1]
    dir_filename = os.path.dirname(filename)
    if workdir is None:
        workdir = common.get_work_dir()

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

    #
    # Find transcoder
    #

    # A limited set of codecs appear to be in the Plex Transcoder 1.19.5
    ffmpeg = common.find_ffmpeg()

    input_info = common.find_input_info(filename)

    if dry_run:
        logger.info(f"{input_info}")

    if filter_skip is None:
        filter_skip = input_info.get(common.K_FORMAT, {}).get(common.K_TAGS, {}).get(common.K_FILTER_SKIP) in ['true',
                                                                                                               'True',
                                                                                                               't',
                                                                                                               'yes',
                                                                                                               'Yes',
                                                                                                               'y', '1']

    streams_file = 0

    #
    # Construct command
    #

    # Global arguments
    arguments = [ffmpeg]
    if not verbose:
        arguments.extend(["-loglevel", "error"])
    arguments.extend(['-hide_banner', '-y', '-analyzeduration', common.ANALYZE_DURATION,
                      '-probesize', common.PROBE_SIZE])

    arguments.extend(["-i", filename])

    subtitle_extract_command = arguments.copy()

    # Load word lists
    censor_list = load_censor_list()
    stop_list = load_stop_list()

    # Compute filter hash
    hash_input = str(FILTER_VERSION) + ','.join(censor_list) + ','.join(stop_list)
    filter_hash = hashlib.sha512(hash_input.encode("utf-8")).hexdigest()
    logger.info(f"expected filter hash = {filter_hash}, expected filter version = {FILTER_VERSION}")

    # Compare and exit if the same
    current_filter_version = input_info.get(common.K_FORMAT, {}).get(common.K_TAGS, {}).get(common.K_FILTER_VERSION)
    current_filter_hash = input_info.get(common.K_FORMAT, {}).get(common.K_TAGS, {}).get(common.K_FILTER_HASH)
    logger.info(f"current filter hash = {current_filter_hash}, current filter version = {current_filter_version}")

    # Find original and filtered subtitle
    subtitle_original, subtitle_filtered, subtitle_filtered_forced = common.find_original_and_filtered_streams(
        input_info,
        common.CODEC_SUBTITLE,
        [common.CODEC_SUBTITLE_ASS,
         common.CODEC_SUBTITLE_SRT,
         common.CODEC_SUBTITLE_SUBRIP],
        language)
    # Find original and filtered audio stream
    audio_original, audio_filtered, _ = common.find_original_and_filtered_streams(input_info,
                                                                                  common.CODEC_AUDIO,
                                                                                  None,
                                                                                  language)

    if filter_skip:
        if audio_filtered is None and subtitle_filtered is None:
            logger.info(f"{filename}: filter skipped due to {common.K_FILTER_SKIP} property")
            return CMD_RESULT_UNCHANGED
        else:
            logger.info(f"{filename}: removing filter due to {common.K_FILTER_SKIP} property")
    else:
        if not force and current_filter_hash == filter_hash and current_filter_version == str(FILTER_VERSION):
            logger.info(f"Stream is already filtered")
            return CMD_RESULT_UNCHANGED

    subtitle_original_bitmap = None
    if not subtitle_original:
        if filter_skip:
            return CMD_RESULT_UNCHANGED
        else:
            subtitle_original_bitmap = ocr_subtitle_bitmap_to_srt(input_info, temp_base, language, verbose=verbose)
            if not subtitle_original_bitmap:
                logger.fatal("Cannot find text based subtitle")
                return CMD_RESULT_ERROR

    if not audio_original:
        if filter_skip:
            return CMD_RESULT_UNCHANGED
        else:
            logger.fatal("Cannot find audio stream for original")
            return CMD_RESULT_ERROR

    tags_filename = f"{temp_base}.tags.xml"
    if subtitle_original_bitmap is not None:
        subtitle_codec = common.CODEC_SUBTITLE_SRT
    else:
        subtitle_codec = subtitle_original.get(common.K_CODEC_NAME)
        if subtitle_codec == common.CODEC_SUBTITLE_SUBRIP:
            subtitle_codec = common.CODEC_SUBTITLE_SRT
    subtitle_original_filename = f"{temp_base}.original.{subtitle_codec}"
    subtitle_filtered_filename = f"{temp_base}.filtered.{subtitle_codec}"
    subtitle_filtered_forced_filename = f"{temp_base}.filtered-forced.{subtitle_codec}"
    subtitle_filtered_previous_filename = f"{temp_base}.filtered.previous.{subtitle_codec}"
    if not debug:
        common.TEMPFILENAMES.append(tags_filename)
        common.TEMPFILENAMES.append(subtitle_original_filename)
        common.TEMPFILENAMES.append(subtitle_filtered_filename)
        common.TEMPFILENAMES.append(subtitle_filtered_forced_filename)
        common.TEMPFILENAMES.append(subtitle_filtered_previous_filename)

    subtitle_original_idx = (subtitle_original or {}).get(common.K_STREAM_INDEX)
    subtitle_filtered_idx = (subtitle_filtered or {}).get(common.K_STREAM_INDEX)
    subtitle_filtered_forced_idx = (subtitle_filtered_forced or {}).get(common.K_STREAM_INDEX)
    audio_original_idx = audio_original.get(common.K_STREAM_INDEX)
    audio_filtered_idx = (audio_filtered or {}).get(common.K_STREAM_INDEX)

    logger.info(f"subtitle original = {subtitle_original_idx}, audio original = {audio_original_idx}")
    logger.info(
        f"subtitle filtered = {subtitle_filtered_idx}, subtitle filtered forced = {subtitle_filtered_forced_idx}, audio filtered = {audio_filtered_idx}")

    if filter_skip:
        if verbose:
            logger.info(f"Removing filtered streams")
        arguments.extend(["-metadata", f"{common.K_FILTER_HASH}="])
        arguments.extend(["-metadata", f"{common.K_FILTER_VERSION}="])
        arguments.extend(["-metadata", f"{common.K_FILTER_STOPPED}="])
        arguments.extend(["-c:s", "copy"])

        # Original audio stream
        arguments.extend(["-map", f"{streams_file}:{audio_original_idx}",
                          "-c:a:0", "copy",
                          "-metadata:s:a:0", f'title={common.TITLE_ORIGINAL}',
                          "-disposition:a:0", "default"])

        audio_output_idx = 1

        # Original subtitle stream
        if subtitle_original_idx is not None:
            arguments.extend(["-map", f"{streams_file}:{subtitle_original_idx}",
                              "-metadata:s:s:0", f'title={common.TITLE_ORIGINAL}',
                              "-disposition:s:0", "default"])
            subtitle_output_idx = 1
        else:
            subtitle_output_idx = 0
    else:
        subtitle_filtered_stream = 1
        arguments.extend(["-i", subtitle_filtered_filename])
        subtitle_filtered_forced_stream = 2
        arguments.extend(["-i", subtitle_filtered_forced_filename])

        if not debug:
            subtitle_extract_command.extend(['-v', 'quiet'])

        if subtitle_original_bitmap is not None:
            subtitle_original_filename = subtitle_original_bitmap
            arguments.extend(["-i", subtitle_original_bitmap])
        else:
            subtitle_extract_command.extend(['-map', f"{streams_file}:{subtitle_original_idx}",
                                             '-c', 'copy', subtitle_original_filename])
        if subtitle_filtered_idx is not None:
            subtitle_extract_command.extend(['-map', f"{streams_file}:{subtitle_filtered_idx}",
                                             '-c', 'copy', subtitle_filtered_previous_filename])
        if subtitle_original_idx is not None or subtitle_filtered_idx is not None:
            if verbose:
                logger.info(f"{common.array_as_command(subtitle_extract_command)}")
            subprocess.run(subtitle_extract_command, check=True)

        # subtitles
        filtered_spans = []
        stopped_spans = []
        if subtitle_codec == common.CODEC_SUBTITLE_ASS:
            ass_data = read_ass(Path(subtitle_original_filename))
            ass_data_forced = copy.copy(ass_data)
            ass_data_forced.events = AssEventList()
            for event in list(ass_data.events):
                original_text = event.text
                filtered_text, stopped = filter_text(censor_list, stop_list, original_text)
                if filtered_text != original_text:
                    event.text = filtered_text
                    ass_data_forced.events.append(copy.copy(event))
                    filtered_spans.append([event.start, event.end])
                    if stopped:
                        stopped_spans.append([event.start, event.end])
            write_ass(ass_data, Path(subtitle_filtered_filename))
            write_ass(ass_data_forced, Path(subtitle_filtered_forced_filename))
        elif subtitle_codec in [common.CODEC_SUBTITLE_SRT, common.CODEC_SUBTITLE_SUBRIP]:
            srt_data = pysrt.open(subtitle_original_filename)
            srt_data_forced = copy.copy(srt_data)
            srt_data_forced.data = []
            for event in list(srt_data):
                original_text = event.text
                filtered_text, stopped = filter_text(censor_list, stop_list, original_text)
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

        tags = input_info[common.K_FORMAT].get(common.K_TAGS, {}).copy()
        tags[common.K_FILTER_HASH] = filter_hash
        tags[common.K_FILTER_VERSION] = FILTER_VERSION
        if len(stopped_spans) > 0:
            tags[common.K_FILTER_STOPPED] = span_list_to_str(stopped_spans)
        common.write_mkv_tags(tags, tags_filename)

        filtered_changed = False
        if subtitle_filtered_idx is not None:
            # We are re-running the filter. Check the previous and current filtered subtitles.
            filtered_changed = not cmp_subtitle_text(subtitle_codec, subtitle_filtered_filename,
                                                     subtitle_filtered_previous_filename)
        else:
            # Not yet filtered, see if we need to filter it
            filtered_changed = len(filtered_spans) > 0
        # If no changes, apply hash attribute
        # If we did OCR on subtitles, don't lose that, transcode to include it without the filtered streams
        if not force and not filtered_changed and not subtitle_original_bitmap and current_filter_version in [
            str(FILTER_VERSION),
            None, '']:
            logger.info(f"No changes, updating filter hash")
            if not dry_run and not debug:
                if verbose:
                    logger.info(mkvpropedit)
                subprocess.run([mkvpropedit, filename, "--tags", f"global:{tags_filename}"])
            return CMD_RESULT_MARKED

        arguments.extend(["-metadata", f"{common.K_FILTER_HASH}={filter_hash}"])
        arguments.extend(["-metadata", f"{common.K_FILTER_VERSION}={FILTER_VERSION}"])
        if len(stopped_spans) > 0:
            arguments.extend(["-metadata", f"{common.K_FILTER_STOPPED}={span_list_to_str(stopped_spans)}"])
        arguments.extend(["-c:s", "copy"])

        # Filtered audio stream
        audio_output_idx = 0
        # TODO: For surround sound, filter only the channel in which we expect speech
        # https://stackoverflow.com/questions/33533401/volume-adjust-and-channel-merge-on-video-using-ffmpeg
        if len(filtered_spans) > 0:
            mute_filters = []
            # TODO: adjust end of span by 0.3 seconds, requires version bump (maybe, might be an artifact of bad cutting that has been fixed)
            for span in filtered_spans:
                if isinstance(span[0], int):
                    mute_filters.append(f"volume=enable='between(t,{span[0] / 1000.0},{span[1] / 1000.0})':volume=0")
                else:
                    mute_filters.append(f"volume=enable='between(t,{span[0] / 1000.0},{span[1] / 1000.0})':volume=0")
            arguments.extend(["-map", f"{streams_file}:{audio_original_idx}",
                              f"-c:a:{audio_output_idx}", "libopus",
                              f"-metadata:s:a:{audio_output_idx}", f'title={common.TITLE_FILTERED}',
                              f"-disposition:a:{audio_output_idx}", "default"])
            common.extend_opus_arguments(arguments, input_info['streams'][audio_original_idx], f'a:{audio_output_idx}',
                                         mute_filters)
            audio_output_idx += 1

        # Original audio stream
        arguments.extend(["-map", f"{streams_file}:{audio_original_idx}",
                          f"-c:a:{audio_output_idx}", "copy",
                          f"-metadata:s:a:{audio_output_idx}", f'title={common.TITLE_ORIGINAL}'])
        if audio_output_idx == 0:
            arguments.extend([f"-disposition:a:{audio_output_idx}", "default"])
        else:
            arguments.extend([f"-disposition:a:{audio_output_idx}", "0"])
        audio_output_idx += 1

        # Filtered subtitle stream
        subtitle_output_idx = 0
        if len(filtered_spans) > 0:
            arguments.extend(["-map", f"{subtitle_filtered_stream}:0",
                              f"-metadata:s:s:{subtitle_output_idx}", f'title={common.TITLE_FILTERED}',
                              f"-metadata:s:s:{subtitle_output_idx}", f'language={language}',
                              f"-disposition:s:{subtitle_output_idx}", "default"])
            subtitle_output_idx += 1
            arguments.extend(["-map", f"{subtitle_filtered_forced_stream}:0",
                              f"-metadata:s:s:{subtitle_output_idx}", f'title={common.TITLE_FILTERED_FORCED}',
                              f"-metadata:s:s:{subtitle_output_idx}", f'language={language}',
                              f"-disposition:s:{subtitle_output_idx}", "-default+forced"])
            subtitle_output_idx += 1

        # Original subtitle stream
        if subtitle_original_idx is not None:
            arguments.extend(["-map", f"{streams_file}:{subtitle_original_idx}"])
        else:
            arguments.extend(["-map", f"{streams_file + 3}:0"])
        arguments.extend([f"-metadata:s:s:{subtitle_output_idx}", f'title={common.TITLE_ORIGINAL}',
                          f"-metadata:s:s:{subtitle_output_idx}", f'language={language}'])
        if subtitle_output_idx == 0:
            arguments.extend([f"-disposition:s:{subtitle_output_idx}", "default"])
        else:
            arguments.extend([f"-disposition:s:{subtitle_output_idx}", "0"])
        subtitle_output_idx += 1

    # Remaining audio streams
    for extra_audio in list(filter(lambda stream: stream[common.K_CODEC_TYPE] == common.CODEC_AUDIO and stream[
        common.K_STREAM_INDEX] not in [audio_original_idx, audio_filtered_idx], input_info['streams'])):
        arguments.extend(["-map", f"{streams_file}:{extra_audio[common.K_STREAM_INDEX]}",
                          f"-c:a:{audio_output_idx}", "copy",
                          f"-disposition:a:{audio_output_idx}", "0"])
        audio_output_idx += 1

    # Remaining subtitle streams
    for extra_audio in list(filter(lambda stream: stream[common.K_CODEC_TYPE] == common.CODEC_SUBTITLE and stream[
        common.K_STREAM_INDEX] not in [subtitle_original_idx, subtitle_filtered_idx, subtitle_filtered_forced_idx],
                                   input_info['streams'])):
        arguments.extend(["-map", f"{streams_file}:{extra_audio[common.K_STREAM_INDEX]}",
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
        has_closed_captions = video_info.get('closed_captions', 0) > 0
        input_video_codec = common.resolve_video_codec(video_info['codec_name'])
        if has_closed_captions:
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
        logger.info(f"{common.array_as_command(arguments)}")
        return CMD_RESULT_FILTERED

    # Check again because there is time between the initial check and when we write to the file
    if common.assert_not_transcoding(input_file, exit=False) != 0:
        return CMD_RESULT_ERROR
    try:
        Path(common.TEMPFILENAME).touch(mode=0o664, exist_ok=False)
    except FileExistsError:
        return CMD_RESULT_ERROR

    logger.info(f"Starting filtering of {filename} to {common.TEMPFILENAME}")
    logger.info(f"{common.array_as_command(arguments)}")
    subprocess.run(arguments, check=True)

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
    result = list(
        loadtxt(os.path.join(os.path.dirname(common.__file__), 'censor_list.txt'), dtype='str', delimiter='\xFF'))
    result = list(filter(phrase_list_accept_condition, result))
    result.sort(key=lambda e: len(re.sub('[^A-Za-z]+', '', e)), reverse=True)
    return result


def load_stop_list():
    # TODO: allow multiple lists based on 'levels' (R, PG, G ?) and allow selection at runtime
    result = loadtxt(os.path.join(os.path.dirname(common.__file__), 'stop_list.txt'), dtype='str', delimiter='\xFF')
    result = list(filter(phrase_list_accept_condition, result))
    return result


def ocr_subtitle_bitmap_to_srt(input_info, temp_base, language=None, verbose=False):
    """
    Attempts to find a bitmap subtitle stream and run OCR to generate an SRT file.
    :return: file name as string or None
    """
    global debug

    ffmpeg = common.find_ffmpeg()
    subtitle_edit = find_subtitle_edit()
    if not subtitle_edit:
        return None
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
        extract_command = [ffmpeg, "-loglevel", "error", '-hide_banner', '-y', '-analyzeduration',
                           common.ANALYZE_DURATION,
                           '-probesize', common.PROBE_SIZE,
                           '-i', input_info['format']['filename'],
                           '-map', f'0:{bluray[common.K_STREAM_INDEX]}', '-c:s', 'copy',
                           subtitle_filename
                           ]
        if verbose:
            logger.info(f"{common.array_as_command(extract_command)}")
        subprocess.run(extract_command, check=True, capture_output=True)

    dvd = find_subtitle_dvdsub(input_info, language)
    if dvd:
        dvd = dvd[0]
        subtitle_filename = f"{temp_base}.vob"
        if not debug:
            common.TEMPFILENAMES.append(subtitle_filename)
        extract_command = [ffmpeg, "-loglevel", "error", '-hide_banner', '-y', '-analyzeduration',
                           common.ANALYZE_DURATION,
                           '-probesize', common.PROBE_SIZE,
                           '-i', input_info['format']['filename'],
                           '-map', f'0:{dvd[common.K_STREAM_INDEX]}', '-c:s', 'dvdsub',
                           subtitle_filename
                           ]
        if verbose:
            logger.info(f"{common.array_as_command(extract_command)}")
        subprocess.run(extract_command, check=True, capture_output=True)

    if subtitle_filename is None:
        return None

    ocr_command = [subtitle_edit, subtitle_filename]
    subprocess.run(ocr_command, check=True)
    if not os.access(subtitle_srt_filename, os.R_OK):
        logger.error(f"SRT not generated from OCR")
        return None

    word_found_pct = words_in_dictionary_pct(subtitle_srt_filename, language)
    if word_found_pct < 93.0:
        logger.error(f"OCR text appears to be incorrect, {word_found_pct}% words found in {language} dictionary")
        return None

    return subtitle_srt_filename


def words_in_dictionary_pct(subtitle_srt_filename, language):
    # verify with spell checker (hunspell) that text looks like English
    if language == common.LANGUAGE_ENGLISH:
        hobj = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
    else:
        hobj = hunspell.HunSpell(f'/usr/share/hunspell/{language[0:2]}.dic', f'/usr/share/hunspell/{language[0:2]}.aff')
    word_count = 0
    word_found_count = 0
    srt_data = pysrt.open(subtitle_srt_filename)
    for event in list(srt_data):
        for word in re.sub('[^A-Za-z\' ]+', ' ', event.text).split():
            word_count += 1
            if hobj.spell(word):
                word_found_count += 1
    word_found_pct = 100.0 * float(word_found_count) / float(word_count)
    logger.info(f"SRT words = {word_count}, found = {word_found_count}, {word_found_pct}%")
    if word_count < 100:
        logger.warning("word count less than 100, returning 0%")
        return 0.0
    return word_found_pct


def find_subtitle_dvdsub(input_info, language=None):
    return one_or_zero_streams(
        common.find_streams_by_codec_and_language(input_info, common.CODEC_SUBTITLE, [common.CODEC_SUBTITLE_DVDSUB],
                                                  language))


def find_subtitle_bluray(input_info, language=None):
    return one_or_zero_streams(
        common.find_streams_by_codec_and_language(input_info, common.CODEC_SUBTITLE, [common.CODEC_SUBTITLE_BLURAY],
                                                  language))


def one_or_zero_streams(streams):
    default_streams = list(
        filter(lambda stream: stream.get('disposition') and stream.get('disposition').get('default') > 0, streams))
    if len(default_streams) > 0:
        return default_streams[0:1]
    return streams[0:1]


def find_subtitle_edit():
    se = which("subtitle-edit")
    local_se = f"{os.getcwd()}/subtitle-edit"
    if not se and os.access(local_se, os.X_OK):
        se = local_se
    if not se:
        logger.fatal("subtitle-edit not found to perform OCR on bitmap subtitles")
        return None
    if not os.access(se, os.X_OK):
        logger.fatal(f"{se} is not an executable")
        return None

    return se


def span_list_to_str(span_list: list) -> str:
    if span_list is None:
        return ''
    return ','.join(
        map(lambda span: common.s_to_ts(span[0] / 1000.0) + '-' + common.s_to_ts(span[1] / 1000.0), span_list))


STOP_CLEAN_PATTERN = "^((?:[{].*?[}])*)"
WORD_BREAKS = r'((?:\\s|[,!]|[{].+?[}]|[\\\\]x[0-9a-fA-F]+|[\\\\][Nh])+|\\b)'
NO_PREVIOUS_WORD = r'(?<!\\w\\s)'
PRE_FILTERED = r'[!@#$%^&*+-]\s?(?:[@#$%^&*+-]\s?){2,}'


def contains_pattern_repl(matchobj):
    result = ''
    # logger.debug(f"{matchobj.groups()}")
    for group_idx in range(1, len(matchobj.groups()) + 1):
        if group_idx % 2 == 0:
            result = result + '***'
        elif matchobj.group(group_idx) is not None:
            result = result + matchobj.group(group_idx)
        pass
    return result


def filter_text(censor_list, stop_list, text) -> (str, bool):
    """
    Filter text using the lists.
    :param censor_list: phrases in the censor list are removed from the text, such as adjectives or exclamations
    :param stop_list: phrases in the stop list indicate suspicious subject and all text is removed
    :param text:
    :return: filtered text, True if phrase in stop list was found
    """
    if re.search(PRE_FILTERED, text, flags=re.IGNORECASE):
        text2 = re.sub(PRE_FILTERED, '***', text)
        if text2 == text:
            # We need the text to change to mute the audio
            text = text2 + ' _'
        else:
            text = text2
    for stop_phrase in stop_list:
        stop_pattern = phrase_to_pattern(stop_phrase)
        if re.search(stop_pattern, text, flags=re.IGNORECASE):
            try:
                text = re.search(STOP_CLEAN_PATTERN, text).expand(r"\1***")
                return text, True
            except re.error as e:
                print(f"ERROR in stop list: {stop_phrase} => {stop_pattern}")
                raise e
    for contains_phrase in censor_list:
        contains_pattern = phrase_to_pattern(contains_phrase)
        # logger.debug(f"{contains_phrase} => {contains_pattern}")
        try:
            text = re.sub(contains_pattern, contains_pattern_repl, text, flags=re.IGNORECASE)
        except re.error as e:
            print(f"ERROR in censor list: {contains_phrase} => {contains_pattern}")
            raise e
    # clean up trailing punctuation
    text = re.sub(r'\*\*\*([,!.?\'’]+)', '***', text)
    # clean up redundant replacements
    text = re.sub(r'(\*\*\*[\s,]*)+\*\*\*', '***', text)
    return text, False


def phrase_to_pattern(phrase):
    """
    Transforms a phrase, which is roughly a regex, into something that will match both SRT and ASS markup.
    """
    phrase_fancy_word_breaks = re.sub(r'([\w\'’]+)\s*', r'(\1)' + WORD_BREAKS, phrase, flags=re.IGNORECASE)
    phrase_beginning_marker = re.sub(r'[\^]', NO_PREVIOUS_WORD, phrase_fancy_word_breaks)
    phrase_starting_word_break = re.sub(r'.', WORD_BREAKS, '.') + phrase_beginning_marker
    return phrase_starting_word_break


def cmp_subtitle_text(subtitle_codec, f1, f2):
    return read_subtitle_text(subtitle_codec, f1) == read_subtitle_text(subtitle_codec, f2)


def read_subtitle_text(subtitle_codec, f):
    lines = []
    if subtitle_codec == common.CODEC_SUBTITLE_ASS:
        ass_data = read_ass(Path(f))
        for event in list(ass_data.events):
            lines.append(event.text)
    elif subtitle_codec in [common.CODEC_SUBTITLE_SRT, common.CODEC_SUBTITLE_SUBRIP]:
        srt_data = pysrt.open(f)
        for event in list(srt_data):
            lines.append(event.text)
    else:
        common.fatal(f"INFO: Unknown subtitle codec {subtitle_codec}")
    return "\n".join(lines)


def phrase_list_accept_condition(e):
    e = e.strip()
    if len(e) == 0:
        return False
    return e[0] != '#'


if __name__ == '__main__':
    common.setup_cli()
    argv = sys.argv[1:]

    dry_run = False
    keep = False
    force = False
    filter_skip = None
    language = common.LANGUAGE_ENGLISH
    workdir = common.get_work_dir()

    try:
        opts, args = getopt.getopt(
            list(argv), "nkdrf",
            ["dry-run", "keep", "debug", "remove", "force", "work-dir="])
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
        profanity_filter(input_file, dry_run=dry_run, keep=keep, force=force, filter_skip=filter_skip, workdir=workdir,
                         verbose=True))
