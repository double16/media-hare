#!/usr/bin/env python3
import atexit
import getopt
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable, Union

import common
from comchap import comchap, write_chapter_metadata, compute_comskip_ini_hash, find_comskip_ini
from common import crop_frame, hwaccel, subtitle, tools, config, constants, edl_util, fsutil
from profanity_filter import MASK_STR

KEYFRAME_DISTANCE_TOLERANCE = 1.0
KEYFRAME_IGNORE_FOR_ENCODING = False
USE_FIRST_KEYFRAME_FOR_START_TIME = True
# http://ffmpeg.org/ffmpeg-all.html#select_002c-aselect
FILTER_AV_CONCAT_DEMUX = False

logger = logging.getLogger(__name__)


#
# Important info on seeking: https://trac.ffmpeg.org/wiki/Seeking
#


def usage():
    print(f"""
Remove commercial from video file using EDL file.
     (If no EDL file is found, comskip will be used to generate one)

Cut start is inclusive, end is exclusive with respect to I-frame. For example, comskip generates the real times of the
commercials regardless of frame type. Cutting without recoding requires cutting at I-frames. Start times not on an
I-frame are backed up to the nearest preceding I-frame. End times not on an I-frame are moved to the next I-frame, but
that I-frame is NOT included in the cut.

When generating cut lists based on I-frame, the start frame must be the first frame of the cut. The end I-frame must be
the first frame of the desired content, i.e. first I-frame AFTER the cut.

NOTE: Times in the EDL are zero based! For example, if you use avidemux to get I-frame times, they are zero based and
should be used. The time markers in the keyframes may not be zero based, they will be adjusted.

Usage: {sys.argv[0]} infile [outfile]
--keep-edl
--keep-meta
--verbose
--comskip-ini=
--work-dir={config.get_work_dir()}
--preset=veryslow,slow,medium,fast,veryfast
    Set ffmpeg preset, defaults to {config.get_global_config_option('ffmpeg', 'preset')}
--force-encode
    Force encoding to be precise, i.e. skip cutting by key frames
--crop-frame
    Detect and crop surrounding frame. Does not modify widescreen formats that have top and bottom frames.
--crop-frame-ntsc
    Detect and crop surrounding frame to one of the NTSC (and HD) common resolutions.
--crop-frame-pal
    Detect and crop surrounding frame to one of the PAL (and HD) common resolutions.
--crop-frame-fixed=w:h[:x:y]
    Crop frame to specified values. If x:y are omitted, the frame is centered.
-v, --vcodec=h264[,hevc,...]
    The video codec: {config.get_global_config_option('video', 'codecs')} (default), h265, mpeg2.
-n, --dry-run
    Dry run
""", file=sys.stderr)


def comcut(infile, outfile, delete_edl=True, force_clear_edl=False, delete_meta=True, verbose=False, debug=False,
           comskipini=None,
           workdir=None, preset=None, hwaccel_requested=None, force_encode=False, dry_run=False,
           crop_frame_op: crop_frame.CropFrameOperation = crop_frame.CropFrameOperation.NONE,
           crop_frame_fixed: Union[str, None] = None,
           desired_video_codecs: Iterable[str] = None):

    if desired_video_codecs is None:
        desired_video_codecs = config.get_global_config_option('video', 'codecs').split(',')

    input_info = common.find_input_info(infile)
    video_info = common.find_video_stream(input_info)
    chapters = input_info.get(constants.K_CHAPTERS, []).copy()
    chapters.sort(key=lambda c: float(c['start_time']))
    if len(list(filter(lambda c: 'Commercial' in c.get("tags", {}).get('title', ''), chapters))) > 0:
        logger.debug(f"ignoring existing chapters {chapters} because they include commercials")
        copychapters = False
        chapters = []
    else:
        copychapters = len(chapters) > 0
    if copychapters:
        logger.debug(f"using existing chapters {chapters}")

    if workdir is None:
        workdir = os.path.dirname(outfile) or '.'

    if not comskipini:
        try:
            comskipini = find_comskip_ini()
        except OSError as e:
            logger.fatal(f"finding {comskipini}", exc_info=e)
            return 255

    if hwaccel_requested is None:
        hwaccel_requested = config.get_global_config_option('ffmpeg', 'hwaccel', fallback=None)

    outextension = outfile.split('.')[-1]
    infile_base = '.'.join(os.path.basename(infile).split('.')[0:-1])

    outfile_base = f".~{infile_base}"
    edlfile = common.edl_for_video(infile)
    metafile = os.path.join(workdir, f"{outfile_base}.ffmeta")
    partsfile = os.path.join(workdir, f"{outfile_base}.parts.txt")

    if not os.access(edlfile, os.R_OK):
        cc_return_code = comchap(infile, outfile, comskipini=comskipini, workdir=workdir, delete_edl=False,
                                 modify_video=False)
        if cc_return_code != 0:
            return cc_return_code

    edl_events = edl_util.parse_edl(edlfile)

    # sanity check edl to ensure it hasn't already been applied, i.e. check if last cut is past duration
    input_duration = float(input_info[constants.K_FORMAT]['duration'])
    if len(edl_events) > 0:
        if edl_events[-1].end > input_duration + 3:
            logger.fatal("edl cuts past end of file")
            return 255

    keyframes = common.load_keyframes_by_seconds(infile)
    logger.debug("Loaded %s keyframes", len(keyframes))

    if USE_FIRST_KEYFRAME_FOR_START_TIME and len(keyframes) > 0:
        start_time = keyframes[0]
    elif video_info and 'start_time' in video_info:
        # The format start_time can be different from the video stream start_time
        start_time = float(video_info['start_time'])
    else:
        start_time = float(input_info[constants.K_FORMAT].get('start_time', 0.0))

    if KEYFRAME_IGNORE_FOR_ENCODING:
        if force_encode or len(
                list(filter(lambda e: e.event_type in [edl_util.EdlType.BACKGROUND_BLUR], edl_events))) > 0:
            keyframes = []
            logger.info("Discarding keyframes because we're encoding")
        else:
            # If we are encoding, we aren't restricted to key frames, so encode if cuts are too far from keyframes
            disable_keyframes = False
            keyframe_distance = 0.0
            for event in list(
                    filter(lambda e: e.event_type in [edl_util.EdlType.CUT, edl_util.EdlType.COMMERCIAL], edl_events)):
                adjusted1 = abs(event.start - common.find_desired_keyframe(keyframes, event.start,
                                                                           common.KeyframeSearchPreference.BEFORE,
                                                                           start_time))
                adjusted2 = abs(
                    event.end - common.find_desired_keyframe(keyframes, event.end, common.KeyframeSearchPreference.AFTER,
                                                             start_time))
                if adjusted1 > KEYFRAME_DISTANCE_TOLERANCE or adjusted2 > KEYFRAME_DISTANCE_TOLERANCE:
                    disable_keyframes = True
                    keyframe_distance = max(adjusted1, adjusted2, keyframe_distance)
            if disable_keyframes:
                logger.info("Re-encoding video due to distance from keyframes: %s seconds", keyframe_distance)
                keyframes = []

    # key is input stream index, value is filename
    subtitle_streams: dict[int, str] = {}
    subtitle_data = {}
    subtitle_muted_streams: list[dict] = []
    # pts is different for video and subtitle, need to always extract subtitles and manage differently
    if len(list(
            filter(lambda e: e.event_type in [edl_util.EdlType.MUTE, edl_util.EdlType.CUT, edl_util.EdlType.COMMERCIAL],
                   edl_events))) > 0:
        # Extract all text based subtitles for masking
        extract_subtitle_command = ['-y', '-i', infile, '-c', 'copy']
        for stream in filter(lambda s: common.is_subtitle_text_stream(s), input_info[constants.K_STREAMS]):
            if stream.get(constants.K_TAGS, {}).get(constants.K_STREAM_TITLE, '') != constants.TITLE_WORDS:
                subtitle_muted_streams.append(stream)
            suffix = stream.get(constants.K_CODEC_NAME)
            if suffix == 'subrip':
                suffix = 'srt'
            temp_fd, subtitle_filename = tempfile.mkstemp(dir=workdir, suffix='.' + suffix)
            os.close(temp_fd)
            if not debug:
                common.TEMPFILENAMES.append(subtitle_filename)
            subtitle_streams[stream[constants.K_STREAM_INDEX]] = subtitle_filename
            extract_subtitle_command.extend(
                ['-map', f"0:{stream[constants.K_STREAM_INDEX]}", subtitle_filename])
        if len(subtitle_streams) > 0:
            logger.debug(tools.ffmpeg.array_as_command(extract_subtitle_command))
            tools.ffmpeg.run(extract_subtitle_command, check=True)
            for stream in filter(lambda s: common.is_subtitle_text_stream(s), input_info[constants.K_STREAMS]):
                subtitle_filename = subtitle_streams[stream[constants.K_STREAM_INDEX]]
                subtitle_data[stream[constants.K_STREAM_INDEX]] = subtitle.read_subtitle_data(
                    stream.get(constants.K_CODEC_NAME),
                    subtitle_filename)

    if delete_meta:
        common.TEMPFILENAMES.append(partsfile)
        common.TEMPFILENAMES.append(metafile)
    if delete_edl and infile == outfile:
        common.TEMPFILENAMES.append(edlfile)

    start = 0
    i = 0
    min_chapter_seconds = 10
    hascommercials = False
    video_filters: list[str] = []
    audio_filters: list[str] = []
    subtitle_filtered = False
    totalcutduration = 0.0
    comskipini_hash = compute_comskip_ini_hash(comskipini, input_info=input_info, workdir=workdir, log_file=False)
    video_encoder_options_tag_value = []

    crop_frame_filter = crop_frame.find_crop_frame_filter(crop_frame_op, input_info, common.get_frame_rate(input_info), crop_frame_fixed)
    if crop_frame_filter:
        video_filters.append(crop_frame_filter)

    # modify subtitle events before cutting
    for edl_event in filter(lambda e: e.event_type == edl_util.EdlType.MUTE, edl_events):
        subtitle_filtered = True
        # subtitle events are in milliseconds
        mute_start = edl_event.start * 1000.0
        mute_end = edl_event.end * 1000.0
        for stream in subtitle_muted_streams:
            data = subtitle_data[stream[constants.K_STREAM_INDEX]]
            if stream[constants.K_CODEC_NAME] == constants.CODEC_SUBTITLE_ASS:
                for event in data.events:
                    if mute_start <= event.end and mute_end >= event.start:
                        logger.debug("Masking subtitle event %s", event)
                        event.set_text(MASK_STR)
            else:
                for event in data:
                    if mute_start <= event.end.ordinal and mute_end >= event.start.ordinal:
                        logger.debug("Masking subtitle event %s", event)
                        event.text = MASK_STR

    with open(metafile, "w") as metafd:
        with open(partsfile, "w") as partsfd:

            metafd.write(f""";FFMETADATA1
[FORMAT]
{constants.K_COMSKIP_HASH}={comskipini_hash}
{constants.K_COMSKIP_SKIP}=true
""")
            for k, v in input_info[constants.K_FORMAT].get(constants.K_TAGS, {}).items():
                if k not in [constants.K_COMSKIP_HASH, constants.K_COMSKIP_SKIP]:
                    metafd.write(f"{k}={v}\n")
            video_title = input_info[constants.K_FORMAT].get(constants.K_STREAM_TITLE, None)
            if video_title:
                metafd.write(f"{constants.K_STREAM_TITLE}={video_title}\n")
            metafd.write("[/FORMAT]\n")

            partsfd.write("ffconcat version 1.0\n")

            for edl_event in edl_events:
                if edl_event.event_type == edl_util.EdlType.BACKGROUND_BLUR:
                    video_filters.append(f"smartblur=lt=30:lr=5.0:enable='between(t,"
                                         f"{edl_event.start - totalcutduration},{edl_event.end - totalcutduration})'")
                    continue
                elif edl_event.event_type == edl_util.EdlType.MUTE:
                    audio_filters.append(f"volume=enable='between(t,"
                                         f"{edl_event.start - totalcutduration},{edl_event.end - totalcutduration})'"
                                         f":volume=0")
                    continue
                elif edl_event.event_type == edl_util.EdlType.SCENE:
                    continue
                elif edl_event.event_type not in [edl_util.EdlType.CUT, edl_util.EdlType.COMMERCIAL]:
                    logger.warning("Unknown EDL type %s, skipping", edl_event.event_type)
                    continue

                end = edl_event.start
                if edl_event.start > 0:  # allow special case of cut from beginning
                    end = common.find_desired_keyframe(keyframes, end, common.KeyframeSearchPreference.BEFORE, start_time)
                    # assert end <= (edl_event.start + start_time)
                    if end != edl_event.start:
                        logger.debug("Moved cut start from %s to keyframe %s", common.s_to_ts(edl_event.start), common.s_to_ts(end))

                start_next = edl_event.end
                if start_next < input_duration:
                    # handle special case of cutting the end
                    if len(keyframes) > 0 and start_next >= (keyframes[-1] + start_time):
                        start_next = input_duration
                    else:
                        start_next = common.find_desired_keyframe(keyframes, start_next,
                                                                  common.KeyframeSearchPreference.AFTER,
                                                                  start_time)
                        # assert start_next >= (edl_event.end + start_time)
                else:
                    # limit to duration
                    start_next = input_duration
                if start_next != edl_event.end:
                    logger.debug("Moved cut end from %s to keyframe %s", common.s_to_ts(edl_event.end), common.s_to_ts(start_next))

                duration = end - start
                if duration > 1:
                    hascommercials = True
                    if copychapters:
                        # these chapters are complete and only need a time shift
                        while len(chapters) > 0 and float(chapters[0]['end_time']) < edl_event.start:
                            i += 1
                            chapter = chapters.pop(0)
                            logger.debug(f"keeping unmodified chapter {chapter}")
                            write_chapter_metadata(metafd,
                                                   max(0.0, float(chapter['start_time']) - totalcutduration),
                                                   float(chapter['end_time']) - totalcutduration,
                                                   chapter.get("tags", {}).get("title", f"Chapter {i}"),
                                                   min_chapter_seconds)
                        # these chapters are cut, completely or partially
                        while len(chapters) > 0 and float(chapters[0]['start_time']) < edl_event.end:
                            chapter = chapters.pop(0)
                            chapter_start = float(chapter['start_time'])
                            chapter_end = float(chapter['end_time'])
                            if chapter_start >= edl_event.start:
                                chapter_start = edl_event.end
                            if chapter_end <= edl_event.end:
                                chapter_end = edl_event.start
                            else:
                                chapter_end -= edl_event.end - edl_event.start
                            if chapter_start >= chapter_end:
                                logger.debug(f"cut chapter {chapter}")
                                continue
                            i += 1
                            logger.debug(f"modified chapter {chapter} to [{chapter_start}, {chapter_end}]")
                            write_chapter_metadata(metafd,
                                                   max(0.0, float(chapter_start) - totalcutduration),
                                                   float(chapter_end) - totalcutduration,
                                                   chapter.get("tags", {}).get("title", f"Chapter {i}"),
                                                   min_chapter_seconds)
                    else:
                        if write_chapter_metadata(metafd,
                                                  start - totalcutduration, end - totalcutduration,
                                                  f"Chapter {i + 1}", min_chapter_seconds):
                            i += 1

                    partsfd.write("file %s\n" % re.sub('([^A-Za-z0-9/])', r'\\\1', os.path.abspath(infile)))
                    partsfd.write(f"inpoint {start}\n")
                    partsfd.write(f"outpoint {end}\n")

                for data in subtitle_data.values():
                    logger.debug("Cutting subtitle (%f, %f)", end - totalcutduration, start_next - totalcutduration)
                    subtitle.subtitle_cut(data, end - totalcutduration, start_next - totalcutduration)

                totalcutduration = totalcutduration + start_next - end
                start = start_next

            # add the final part from last commercial to end of file
            end = input_duration
            duration = end - start
            if duration > 1:
                hascommercials = True
                logger.debug("Including last %f seconds, start = %f, end = %f", duration, start, end)
                if copychapters:
                    while len(chapters) > 0:
                        i += 1
                        chapter = chapters.pop(0)
                        logger.debug(f"(last cut) keeping unmodified chapter {chapter}")
                        write_chapter_metadata(metafd,
                                               max(0.0, float(chapter['start_time']) - totalcutduration),
                                               float(chapter['end_time']) - totalcutduration,
                                               chapter.get("tags", {}).get("title", f"Chapter {i}"),
                                               min_chapter_seconds)
                else:
                    if write_chapter_metadata(metafd, start - totalcutduration, end - totalcutduration,
                                              f"Chapter {i + 1}",
                                              min_chapter_seconds):
                        i += 1
                partsfd.write("file %s\n" % re.sub('([^A-Za-z0-9/])', r'\\\1', os.path.abspath(infile)))
                partsfd.write(f"inpoint {start}\n")

    if hascommercials or len(video_filters) > 0 or len(audio_filters) > 0 or subtitle_filtered:
        # doing it this way to keep the ident level one less
        pass
    else:
        logger.debug("Nothing found to change")
        if infile == outfile:
            return 0
        else:
            # we are not creating a new outfile, so don't return success
            return 1

    if FILTER_AV_CONCAT_DEMUX:
        # not really sure how this affects subtitle streams
        audio_filters.append('aselect=concatdec_select')
        video_filters.append('select=concatdec_select')

    hwaccel.hwaccel_configure(hwaccel_requested)

    ffmpeg_command = []
    ffmpeg_command.extend(hwaccel.hwaccel_threads())

    if not debug:
        ffmpeg_command.append("-hide_banner")

    ffmpeg_command.extend(["-nostdin"])

    height = common.get_video_height(video_info)
    depth = common.get_video_depth(video_info)
    target_video_codec = common.resolve_video_codec(desired_video_codecs, height, video_info)

    if len(video_filters) > 0 or len(keyframes) == 0 or force_encode:
        input_video_codec = common.resolve_video_codec(video_info[constants.K_CODEC_NAME])
        ffmpeg_command.extend(
            hwaccel.hwaccel_prologue(input_video_codec=input_video_codec, target_video_codec=target_video_codec))
        ffmpeg_command.extend(hwaccel.hwaccel_decoding(input_video_codec))

    ffmpeg_command.extend(["-i", metafile,
                           "-f", "concat", "-safe", "0", "-segment_time_metadata", "1", "-i", partsfile])

    input_stream_idx = 2
    subtitle_stream_idx_to_input_idx: dict[int, int] = {}
    if len(subtitle_data) > 0:
        for stream in filter(lambda s: common.is_subtitle_text_stream(s), input_info[constants.K_STREAMS]):
            data = subtitle_data[stream[constants.K_STREAM_INDEX]]
            subtitle_filename = subtitle_streams[stream[constants.K_STREAM_INDEX]]
            logger.debug("Writing subtitle %s (%s) to %s",
                         stream.get(constants.K_TAGS, {}).get(constants.K_STREAM_TITLE, ''),
                         stream[constants.K_CODEC_NAME],
                         subtitle_filename
                         )
            subtitle.write_subtitle_data(stream[constants.K_CODEC_NAME], subtitle_filename, data)
            if os.path.getsize(subtitle_filename) == 0:
                logger.info("Subtitle %s is empty", subtitle_filename)
            else:
                ffmpeg_command.extend(["-i", subtitle_filename])
                subtitle_stream_idx_to_input_idx[stream[constants.K_STREAM_INDEX]] = input_stream_idx
                input_stream_idx += 1

    ffmpeg_command.extend(['-max_muxing_queue_size', '1024',
                           '-async', '1',
                           '-max_interleave_delta', '0',
                           "-avoid_negative_ts", "1",
                           "-map_metadata", "0"])

    # filters will re-order output streams, so we need to map each individually
    output_file = 1
    output_stream_idx = 0

    for stream in common.sort_streams(input_info[constants.K_STREAMS]):
        if common.is_video_stream(stream) and (len(video_filters) > 0 or len(keyframes) == 0 or force_encode):
            height = common.get_video_height(stream)
            crf, bitrate, qp = common.recommended_video_quality(height, target_video_codec, depth)

            # adjust frame rate
            desired_frame_rate = config.get_global_config_frame_rate('post_process', 'frame_rate', None)
            if desired_frame_rate is not None:
                desired_frame_rate = constants.FRAME_RATE_NAMES.get(desired_frame_rate.lower(), desired_frame_rate)
                if common.should_adjust_frame_rate(current_frame_rate=stream['avg_frame_rate'],
                                                   desired_frame_rate=desired_frame_rate,
                                                   tolerance=0.05):
                    video_filters.append(common.fps_video_filter(desired_frame_rate))

            encoding_options, encoding_method = hwaccel.hwaccel_encoding(output_stream=str(output_stream_idx),
                                                                         codec=target_video_codec, output_type="mkv",
                                                                         tune=None, preset=preset, crf=crf, qp=qp,
                                                                         target_bitrate=bitrate, bit_depth=depth)

            # add common video filters if we are doing filtering
            video_filters.append('format=nv12')
            if hwaccel.hwaccel_required_hwupload_filter():
                video_filters.append('hwupload')

            video_filter_str = f"[{output_file}:{str(stream[constants.K_STREAM_INDEX])}]yadif"
            for idx, value in enumerate(video_filters):
                video_filter_str += f"[{idx}];[{idx}]"
                video_filter_str += value
            output_mapping = f"[v{output_stream_idx}]"
            video_filter_str += output_mapping

            ffmpeg_command.extend(["-filter_complex", video_filter_str])
            ffmpeg_command.extend(["-map", output_mapping])

            ffmpeg_command.extend(encoding_options)
            video_encoder_options_tag_value.extend(encoding_options)
        elif common.is_audio_stream(stream) and len(audio_filters) > 0:
            common.map_opus_audio_stream(ffmpeg_command, stream, output_file, str(output_stream_idx), audio_filters)
        elif common.is_subtitle_text_stream(stream) and len(subtitle_data) > 0:
            if stream[constants.K_STREAM_INDEX] not in subtitle_stream_idx_to_input_idx:
                continue
            ffmpeg_command.extend([
                "-map", f"{subtitle_stream_idx_to_input_idx[stream[constants.K_STREAM_INDEX]]}:0",
                f"-c:{output_stream_idx}", "copy"])
            if constants.K_STREAM_TITLE in stream[constants.K_TAGS]:
                ffmpeg_command.extend(
                    [f"-metadata:s:{output_stream_idx}", f'title={stream[constants.K_TAGS][constants.K_STREAM_TITLE]}'])
            if constants.K_STREAM_LANGUAGE in stream[constants.K_TAGS]:
                ffmpeg_command.extend(
                    [f"-metadata:s:{output_stream_idx}",
                     f'language={stream[constants.K_TAGS][constants.K_STREAM_LANGUAGE]}'])
        else:
            ffmpeg_command.extend(["-map", f"{output_file}:{str(stream[constants.K_STREAM_INDEX])}"])
            ffmpeg_command.extend([f"-c:{output_stream_idx}", "copy"])

        # the concat demuxer sets all streams to default
        dispositions = []
        if stream.get(constants.K_DISPOSITION, {}).get('default', 0) == 1:
            dispositions.append("default")
        if stream.get(constants.K_DISPOSITION, {}).get('forced', 0) == 1:
            dispositions.append("forced")
        if len(dispositions) == 0:
            dispositions.append("0")
        ffmpeg_command.extend([f"-disposition:{output_stream_idx}", ",".join(dispositions)])

        output_stream_idx += 1

    if len(video_encoder_options_tag_value) > 0:
        ffmpeg_command.extend(
            ['-metadata', f"{constants.K_ENCODER_OPTIONS}={' '.join(video_encoder_options_tag_value)}"])
    if common.should_replace_media_title(input_info):
        ffmpeg_command.extend(
            ['-metadata', f"{constants.K_MEDIA_TITLE}={common.get_media_title_from_filename(input_info)}"])
    ffmpeg_command.extend(['-metadata', f"{constants.K_MEDIA_PROCESSOR}={constants.V_MEDIA_PROCESSOR}"])

    ffmpeg_command.append('-y')

    temp_outfile = None
    if infile == outfile:
        temp_fd, temp_outfile = tempfile.mkstemp(prefix='.~', suffix='.' + outextension, dir=os.path.dirname(infile))
        os.close(temp_fd)
        ffmpeg_command.append(temp_outfile)
    else:
        ffmpeg_command.append(outfile)

    if dry_run:
        logger.info(tools.ffmpeg.array_as_command(ffmpeg_command))
        return 0
    else:
        logger.debug(tools.ffmpeg.array_as_command(ffmpeg_command))

    try:
        tools.ffmpeg.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        with open(partsfile, "r") as f:
            print(f.read(), file=sys.stderr)
        print(tools.ffmpeg.array_as_command(ffmpeg_command), file=sys.stderr)
        raise e

    # verify video is valid
    try:
        common.find_input_info(temp_outfile or outfile)
    except Exception:
        logger.error("Cut file is not readable by ffmpeg, skipping")
        os.remove(temp_outfile or outfile)
        return 255

    if temp_outfile is not None:
        shutil.move(temp_outfile, outfile)

    fsutil.match_owner_and_perm(target_path=outfile, source_path=infile)

    if force_clear_edl or (not delete_edl and infile == outfile):
        # change EDL file to match cut
        edlfiles = [edlfile]
        if '.bak.edl' not in edlfile:
            edlfiles.append(common.replace_extension(edlfile, 'bak.edl'))
        for fn in edlfiles:
            if os.path.isfile(fn):
                with open(fn, "w") as f:
                    f.write("## cut complete v2\n")

    # remove files that are invalidated because of cutting
    for ext in ['csv', 'txt', 'log']:
        extfile = common.replace_extension(infile, ext)
        if os.path.exists(extfile):
            os.remove(extfile)

    return 0


def comcut_cli(argv):
    delete_edl = not config.get_global_config_boolean('general', 'keep_edl')
    delete_meta = not config.get_global_config_boolean('general', 'keep_meta')
    no_curses = False
    verbose = False
    debug = False
    comskipini = None
    workdir = config.get_work_dir()
    preset = None
    force_encode = False
    dry_run = False
    crop_frame_op = crop_frame.CropFrameOperation.NONE
    crop_frame_fixed = None
    desired_video_codecs = None

    dvrconfig = list(
        filter(lambda e: "crop-frame" in e or "preset" in e, common.get_arguments_from_config(argv, '.dvrconfig')))

    try:
        opts, args = getopt.getopt(dvrconfig + list(argv), "pnv:",
                                   ["keep-edl", "keep-meta", "verbose", "debug", "comskip-ini=", "work-dir=",
                                    "preset=", "force-encode", "dry-run", "crop-frame", "crop-frame-ntsc",
                                    "crop-frame-pal", "crop-frame-fixed=", "vcodec=", "no-curses"])
    except getopt.GetoptError:
        usage()
        return 255
    for opt, arg in opts:
        if opt == '--help':
            usage()
            return 255
        elif opt == "--keep-edl":
            delete_edl = False
        elif opt == "--keep-meta":
            delete_meta = False
        elif opt == "--verbose":
            verbose = True
            logging.getLogger().setLevel(logging.DEBUG)
        elif opt == "--no-curses":
            no_curses = True
        elif opt == "--debug":
            debug = True
        elif opt == "--comskip-ini":
            comskipini = arg
        elif opt == "--work-dir":
            workdir = arg
        elif opt == "--force-encode":
            force_encode = True
        elif opt in ("-p", "--preset"):
            preset = arg
        elif opt in ("-n", "--dry-run"):
            dry_run = True
            no_curses = True
        elif opt == "--crop-frame":
            crop_frame_op = crop_frame.CropFrameOperation.DETECT
        elif opt == "--crop-frame-ntsc":
            crop_frame_op = crop_frame.CropFrameOperation.NTSC
        elif opt == "--crop-frame-pal":
            crop_frame_op = crop_frame.CropFrameOperation.PAL
        elif opt == "--crop-frame-fixed":
            crop_frame_op = crop_frame.CropFrameOperation.FIXED
            crop_frame_fixed = arg
        elif opt in ("-v", "--vcodec"):
            desired_video_codecs = arg.split(',')

    if not args:
        usage()
        return 255

    if not preset or preset == 'copy':
        preset = os.environ.get('PRESET', config.get_global_config_option('ffmpeg', 'preset'))

    common.cli_wrapper(comcut_cli_run, args=args, delete_edl=delete_edl, delete_meta=delete_meta, verbose=verbose,
                       debug=debug, comskipini=comskipini, workdir=workdir, preset=preset,
                       force_encode=force_encode, dry_run=dry_run, crop_frame_op=crop_frame_op,
                       crop_frame_fixed=crop_frame_fixed,
                       desired_video_codecs=desired_video_codecs, no_curses=no_curses)


def comcut_cli_run(args: list, delete_edl, delete_meta, verbose, debug, comskipini, workdir, preset, force_encode,
                   dry_run, crop_frame_op, crop_frame_fixed, desired_video_codecs):
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    atexit.register(common.finish)

    return_code = 0
    for infile, outfile in common.generate_video_files(args, fail_on_missing=True):
        this_file_return_code = comcut(infile, outfile, delete_edl=delete_edl, delete_meta=delete_meta, verbose=verbose,
                                       debug=debug, comskipini=comskipini, workdir=workdir, preset=preset,
                                       force_encode=force_encode, dry_run=dry_run, crop_frame_op=crop_frame_op,
                                       crop_frame_fixed=crop_frame_fixed,
                                       desired_video_codecs=desired_video_codecs)
        if this_file_return_code != 0 and return_code == 0:
            return_code = this_file_return_code

    return return_code


if __name__ == '__main__':
    os.nice(12)
    sys.exit(comcut_cli(sys.argv[1:]))
