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

import common
from comchap import comchap, write_chapter_metadata, compute_comskip_ini_hash, find_comskip_ini
from profanity_filter import MASK_STR
from common import subtitle
from common import hwaccel

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
--work-dir={common.get_work_dir()}
--preset=veryslow,slow,medium,fast,veryfast
    Set ffmpeg preset, defaults to {common.get_global_config_option('ffmpeg', 'preset')}
""", file=sys.stderr)


@common.finisher
def comcut(infile, outfile, delete_edl=True, force_clear_edl=False, delete_meta=True, verbose=False, debug=False,
           comskipini=None,
           workdir=None, preset=None, hwaccel_requested=None):
    ffmpeg = common.find_ffmpeg()

    input_info = common.find_input_info(infile)
    chapters = input_info.get(common.K_CHAPTERS, []).copy()
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
        hwaccel_requested = common.get_global_config_option('ffmpeg', 'hwaccel', fallback=None)

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

    edl_events = common.parse_edl(edlfile)

    # sanity check edl to ensure it hasn't already been applied, i.e. check if last cut is past duration
    input_duration = float(input_info[common.K_FORMAT]['duration'])
    if len(edl_events) > 0:
        if edl_events[-1].end > input_duration + 1:
            logger.fatal(f"edl cuts past end of file")
            return 255

    keyframes = []
    if len(list(filter(lambda e: e.event_type in [common.EdlType.BACKGROUND_BLUR], edl_events))) == 0:
        # If we are recoding, we aren't restricted to key frames
        keyframes = common.load_keyframes_by_seconds(infile)
        logger.debug("Loaded %s keyframes", len(keyframes))

    # key is input stream index, value is filename
    subtitle_streams: dict[int, str] = {}
    subtitle_data = {}
    if len(list(filter(lambda e: e.event_type in [common.EdlType.MUTE], edl_events))) > 0:
        # Extract all of the text based subtitles for masking
        extract_subtitle_command = [common.find_ffmpeg(), '-y', '-i', infile, '-c', 'copy']
        for stream in filter(lambda s: common.is_subtitle_text_stream(s), input_info[common.K_STREAMS]):
            suffix = stream.get(common.K_CODEC_NAME)
            if suffix == 'subrip':
                suffix = 'srt'
            temp_fd, subtitle_filename = tempfile.mkstemp(dir=workdir, suffix='.' + suffix)
            os.close(temp_fd)
            if not debug:
                common.TEMPFILENAMES.append(subtitle_filename)
            subtitle_streams[stream[common.K_STREAM_INDEX]] = subtitle_filename
            extract_subtitle_command.extend(
                ['-map', f"0:{stream[common.K_STREAM_INDEX]}", subtitle_filename])
        if len(subtitle_streams) > 0:
            logger.debug(common.array_as_command(extract_subtitle_command))
            subprocess.run(extract_subtitle_command, check=True, capture_output=not verbose)
            for stream in filter(lambda s: common.is_subtitle_text_stream(s), input_info[common.K_STREAMS]):
                subtitle_filename = subtitle_streams[stream[common.K_STREAM_INDEX]]
                subtitle_data[stream[common.K_STREAM_INDEX]] = subtitle.read_subtitle_data(
                    stream.get(common.K_CODEC_NAME),
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
    subtitle_filters: list[(float, float)] = []
    totalcutduration = 0.0
    comskipini_hash = compute_comskip_ini_hash(comskipini, input_info=input_info, workdir=workdir)
    video_encoder_options_tag_value = []

    with open(metafile, "w") as metafd:
        with open(partsfile, "w") as partsfd:

            metafd.write(f""";FFMETADATA1
[FORMAT]
{common.K_COMSKIP_HASH}={comskipini_hash}
""")
            for k, v in input_info[common.K_FORMAT].get(common.K_TAGS, {}).items():
                if k != common.K_COMSKIP_HASH:
                    metafd.write(f"{k}={v}\n")
            video_title = input_info[common.K_FORMAT].get(common.K_STREAM_TITLE, None)
            if video_title:
                metafd.write(f"{common.K_STREAM_TITLE}={video_title}\n")
            metafd.write("[/FORMAT]\n")

            partsfd.write("ffconcat version 1.0\n")

            for edl_event in edl_events:
                if edl_event.event_type == common.EdlType.BACKGROUND_BLUR:
                    video_filters.append(f"smartblur=lt=30:lr=5.0:enable='between(t,"
                                         f"{edl_event.start - totalcutduration},{edl_event.end - totalcutduration})'")
                    continue
                elif edl_event.event_type == common.EdlType.MUTE:
                    audio_filters.append(f"volume=enable='between(t,"
                                         f"{edl_event.start - totalcutduration},{edl_event.end - totalcutduration})'"
                                         f":volume=0")
                    # subtitle events are in milliseconds
                    subtitle_filters.append(((edl_event.start - totalcutduration) * 1000.0, (edl_event.end- totalcutduration) * 1000.0))
                    continue
                elif edl_event.event_type == common.EdlType.SCENE:
                    continue
                elif edl_event.event_type not in [common.EdlType.CUT, common.EdlType.COMMERCIAL]:
                    logger.warning("Unknown EDL type %s, skipping", edl_event.event_type)
                    continue

                end = edl_event.start
                end = common.find_desired_keyframe(keyframes, end, common.KeyframeSearchPreference.BEFORE)
                if end != edl_event.start:
                    logger.debug("Moved cut start from %f to keyframe %f", edl_event.start, end)

                start_next = edl_event.end
                if start_next < input_duration:
                    # handle special case of cutting the end
                    if len(keyframes) > 0 and abs(start_next - (keyframes[-1] - keyframes[0])) < 0.05:
                        start_next = input_duration
                    else:
                        start_next = common.find_desired_keyframe(keyframes, start_next,
                                                                  common.KeyframeSearchPreference.AFTER)
                else:
                    # limit to duration
                    start_next = input_duration
                if start_next != edl_event.end:
                    logger.debug("Moved cut end from %f to keyframe %f", edl_event.end, start_next)

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

    if hascommercials or len(video_filters) > 0 or len(audio_filters) > 0 or len(subtitle_filters) > 0:
        # doing it this way to keep the ident level one less
        pass
    else:
        logger.debug("Nothing found to change")
        if infile == outfile:
            return 0
        else:
            # we are not creating a new outfile, so don't return success
            return 1

    hwaccel.hwaccel_configure(hwaccel_requested)

    ffmpeg_command = [ffmpeg]
    ffmpeg_command.extend(hwaccel.hwaccel_threads())

    if not verbose:
        ffmpeg_command.extend(["-loglevel", "error"])
    if not debug:
        ffmpeg_command.append("-hide_banner")

    ffmpeg_command.extend(["-nostdin", "-i", metafile,
                           "-f", "concat", "-safe", "0", "-i", partsfile])

    input_stream_idx = 2
    subtitle_stream_idx_to_input_idx: dict[int, int] = {}
    if len(subtitle_filters) > 0:
        # Mask the subtitles for the muted ranges
        logger.debug("subtitle_filters = %s", subtitle_filters)
        for stream in filter(lambda s: common.is_subtitle_text_stream(s), input_info[common.K_STREAMS]):
            data = subtitle_data[stream[common.K_STREAM_INDEX]]
            if stream[common.K_CODEC_NAME] == common.CODEC_SUBTITLE_ASS:
                for subtitle_filter in subtitle_filters:
                    for event in data.events:
                        if subtitle_filter[0] <= event.end and subtitle_filter[1] >= event.start:
                            logger.debug("Masking subtitle event %s", event)
                            event.text = MASK_STR
            else:
                for subtitle_filter in subtitle_filters:
                    for event in data:
                        if subtitle_filter[0] <= event.end.ordinal and subtitle_filter[1] >= event.start.ordinal:
                            logger.debug("Masking subtitle event %s", event)
                            event.text = MASK_STR
            subtitle_filename = subtitle_streams[stream[common.K_STREAM_INDEX]]
            subtitle.write_subtitle_data(stream[common.K_CODEC_NAME], subtitle_filename, data)
            ffmpeg_command.extend(["-i", subtitle_filename])
            subtitle_stream_idx_to_input_idx[stream[common.K_STREAM_INDEX]] = input_stream_idx
            input_stream_idx += 1

    ffmpeg_command.extend(['-max_muxing_queue_size', '1024',
                           '-async', '1',
                           '-max_interleave_delta', '0',
                           "-avoid_negative_ts", "1",
                           "-map_metadata", "0"])

    # filters will re-order output streams so we need to map each individually
    output_file = 1
    output_stream_idx = 0

    for stream in common.sort_streams(input_info[common.K_STREAMS]):
        if common.is_video_stream(stream) and len(video_filters) > 0:
            input_video_codec = common.resolve_video_codec(stream[common.K_CODEC_NAME])
            ffmpeg_command.extend(hwaccel.hwaccel_prologue(input_video_codec=input_video_codec, target_video_codec=input_video_codec))
            ffmpeg_command.extend(hwaccel.hwaccel_decoding(input_video_codec))
            ffmpeg_command.extend(["-map", f"{output_file}:{str(stream[common.K_STREAM_INDEX])}"])
            height = common.get_video_height(stream)
            crf, bitrate, qp = common.recommended_video_quality(height, input_video_codec)

            # adjust frame rate
            desired_frame_rate = common.get_global_config_frame_rate('post_process', 'frame_rate', None)
            if desired_frame_rate is not None:
                desired_frame_rate = common.FRAME_RATE_NAMES.get(desired_frame_rate.lower(), desired_frame_rate)
                if common.should_adjust_frame_rate(current_frame_rate=stream['avg_frame_rate'],
                                                   desired_frame_rate=desired_frame_rate,
                                                   tolerance=0.05):
                    video_filters.append(common.fps_video_filter(desired_frame_rate))

            ffmpeg_command.extend([f"-filter_complex:{output_stream_idx}", ",".join(video_filters)])

            encoding_options, encoding_method = hwaccel.hwaccel_encoding(output_stream=str(output_stream_idx),
                                                                         codec=input_video_codec, output_type="mkv",
                                                                         tune=None, preset=preset, crf=crf, qp=qp,
                                                                         target_bitrate=bitrate)

            ffmpeg_command.extend(encoding_options)
            video_encoder_options_tag_value.extend(encoding_options)
        elif common.is_audio_stream(stream) and len(audio_filters) > 0:
            ffmpeg_command.extend(["-map", f"{output_file}:{str(stream[common.K_STREAM_INDEX])}"])
            # Preserve original audio codec??
            ffmpeg_command.extend([f"-c:{output_stream_idx}", hwaccel.ffmpeg_sw_codec("opus")])
            common.extend_opus_arguments(ffmpeg_command, stream, str(output_stream_idx), audio_filters)
        elif common.is_subtitle_text_stream(stream) and len(subtitle_filters) > 0:
            ffmpeg_command.extend([
                "-map", f"{subtitle_stream_idx_to_input_idx[stream[common.K_STREAM_INDEX]]}:0",
                f"-c:{output_stream_idx}", "copy"])
            if common.K_STREAM_TITLE in stream[common.K_TAGS]:
                ffmpeg_command.extend(
                    [f"-metadata:s:{output_stream_idx}", f'title={stream[common.K_TAGS][common.K_STREAM_TITLE]}'])
            if common.K_STREAM_LANGUAGE in stream[common.K_TAGS]:
                ffmpeg_command.extend(
                    [f"-metadata:s:{output_stream_idx}", f'language={stream[common.K_TAGS][common.K_STREAM_LANGUAGE]}'])
        else:
            ffmpeg_command.extend(["-map", f"{output_file}:{str(stream[common.K_STREAM_INDEX])}"])
            ffmpeg_command.extend([f"-c:{output_stream_idx}", "copy"])

        # the concat demuxer sets all streams to default
        dispositions = []
        if stream.get(common.K_DISPOSITION, {}).get('default', 0) == 1:
            dispositions.append("default")
        if stream.get(common.K_DISPOSITION, {}).get('forced', 0) == 1:
            dispositions.append("forced")
        if len(dispositions) == 0:
            dispositions.append("0")
        ffmpeg_command.extend([f"-disposition:{output_stream_idx}", ",".join(dispositions)])

        output_stream_idx += 1

    if len(video_encoder_options_tag_value) > 0:
        ffmpeg_command.extend(['-metadata', f"{common.K_ENCODER_OPTIONS}={' '.join(video_encoder_options_tag_value)}"])

    ffmpeg_command.append('-y')

    temp_outfile = None
    if infile == outfile:
        temp_fd, temp_outfile = tempfile.mkstemp(suffix='.' + outextension, dir=workdir)
        os.close(temp_fd)
        ffmpeg_command.append(temp_outfile)
    else:
        ffmpeg_command.append(outfile)
    logger.debug(common.array_as_command(ffmpeg_command))

    try:
        subprocess.run(ffmpeg_command, check=True, capture_output=not verbose)
    except subprocess.CalledProcessError as e:
        with open(partsfile, "r") as f:
            print(f.read())
        raise e

    # verify video is valid
    try:
        common.find_input_info(temp_outfile or outfile)
    except:
        logger.error(f"Cut file is not readable by ffmpeg, skipping")
        os.remove(temp_outfile or outfile)
        return 255

    if temp_outfile is not None:
        shutil.move(temp_outfile, outfile)

    common.match_owner_and_perm(target_path=outfile, source_path=infile)

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
    delete_edl = not common.get_global_config_boolean('general', 'keep_edl')
    delete_meta = not common.get_global_config_boolean('general', 'keep_meta')
    verbose = False
    debug = False
    comskipini = None
    workdir = common.get_work_dir()
    preset = None

    try:
        opts, args = getopt.getopt(list(argv), "p",
                                   ["keep-edl", "keep-meta", "verbose", "debug", "comskip-ini=", "work-dir=",
                                    "preset="])
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
        elif opt == "--debug":
            debug = True
        elif opt == "--comskip-ini":
            comskipini = arg
        elif opt == "--work-dir":
            workdir = arg
        elif opt in ("-p", "--preset"):
            preset = arg

    if not args:
        usage()
        return 255

    if not preset:
        preset = os.environ.get('PRESET', common.get_global_config_option('ffmpeg', 'preset'))

    atexit.register(common.finish)

    return_code = 0
    for infile, outfile in common.generate_video_files(args):
        this_file_return_code = comcut(infile, outfile, delete_edl=delete_edl, delete_meta=delete_meta, verbose=verbose,
                                       debug=debug, comskipini=comskipini, workdir=workdir, preset=preset)
        if this_file_return_code != 0 and return_code == 0:
            return_code = this_file_return_code

    return return_code


if __name__ == '__main__':
    os.nice(12)
    common.setup_cli()
    sys.exit(comcut_cli(sys.argv[1:]))
