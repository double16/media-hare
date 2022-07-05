#!/usr/bin/env python3

import atexit
import getopt
import logging
import math
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from tempfile import mkstemp

import common
from profanity_filter import do_profanity_filter

SHORT_VIDEO_SECONDS = 30
FRAME_RATE_NAMES = {'ntsc': '30000/1001', 'pal': '25.0', 'film': '24.0', 'ntsc_film': '24000/1001'}

logger = logging.getLogger(__name__)


def usage():
    print(f"""Plex DVR Post Processing Script

Transcode input file to .mkv. Will extract subtitle stream from closed caption.

Hardware transcoding: https://trac.ffmpeg.org/wiki/Hardware/VAAPI
Experience with hardware encoding is poor quality and corruption

A file in the same or parent directories of the input file named '.dvrconfig' can contain command line arguments.
The file closest to the input file will be taken. Comments start with '#'.

{sys.argv[0]} [options] file

--dry-run
    Output command that would be used but do nothing.
--keep
    Keep original file in a backup prefixed by ".~".
--vcodec=h264[,hvec,...]
    The video codec: {common.get_global_config_option('video', 'codecs')} (default), h265, mpeg2.
--acodec=opus[,aac,...]
    The audio codec: {common.get_global_config_option('audio', 'codecs')} (default), aac, ac3, ...
--height=480
    Scale down to this height, maintaining aspect ratio.
--output-type=mkv
    Output type: mkv (default), ts, mp4, ...
--prevent-larger=true,false
    Prevent conversion to a larger file (default is {common.get_global_config_boolean('post_process', 'prevent_larger')}).
--hwaccel=false,auto,full
    Enable hardware acceleration, if available (default is {common.get_global_config_option('ffmpeg', 'hwaccel')}).
--stereo
    Scale down audio to stereo.
--preset=copy,medium,fast,veryfast
    Set ffmpeg preset, with a special "copy" for only copying streams.
--tune=[animation,...]
    Set ffmpeg tune value.
--rerun
    Re-encode streams already in the desired codec.
--profanity-filter
    Include profanity filter in output.
--crop-frame
    Detect and crop surrounding frame. Does not modify widescreen formats that have top and bottom frames.
""", file=sys.stderr)


def find_mount_point(path):
    path = os.path.realpath(path)
    orig_dev = os.stat(path).st_dev

    while path != '/':
        dir = os.path.dirname(path)
        if os.stat(dir).st_dev != orig_dev:
            # we crossed the device border
            break
        path = dir
    return path


def has_hw_codec(ffmpeg, vainfo, codec):
    if not os.access(vainfo, os.X_OK):
        return False

    if f"{codec}_vaapi" not in subprocess.check_output([ffmpeg, "-codecs"], stderr=subprocess.STDOUT, text=True):
        return False

    if codec == 'mpeg2' or codec == 'mpeg2video':
        vaprofile = 'VAProfileMPEG2Main'
    else:
        vaprofile = f"VAProfileH{codec.upper()}High"

    return vaprofile in subprocess.check_output([vainfo, "-a"], stderr=subprocess.STDOUT, text=True)


def parse_args(argv) -> (list[str], dict):
    desired_video_codecs = None
    desired_audio_codecs = None
    desired_frame_rate = None
    desired_height = None
    preset = None
    prevent_larger_file = common.get_global_config_boolean('post_process', 'prevent_larger')
    output_type = "mkv"
    tune = None
    hwaccel_requested = common.get_global_config_option('ffmpeg', 'hwaccel', fallback=None)
    dry_run = False
    keep = False
    stereo = False
    rerun = None
    ignore_errors = None
    profanity_filter = common.get_global_config_boolean('post_process', 'profanity_filter')
    crop_frame = False

    try:
        opts, args = getopt.getopt(common.get_arguments_from_config(argv, '.dvrconfig') + list(argv),
                                   "v:a:h:o:t:p:f:w:nklsrc",
                                   ["vcodec=", "acodec=", "height=", "output-type=", "tune=", "preset=", "framerate=",
                                    "hwaccel=", "dry-run", "keep",
                                    "prevent-larger=", "stereo", "rerun", "no-rerun", "ignore-errors",
                                    "profanity-filter", "crop-frame"])
    except getopt.GetoptError:
        return None
    for opt, arg in opts:
        if opt == '--help':
            return None
        elif opt in ("-v", "--vcodec"):
            desired_video_codecs = arg.split(',')
        elif opt in ("-a", "--acodec"):
            desired_audio_codecs = arg.split(',')
        elif opt in ("-h", "--height"):
            desired_height = int(arg)
        elif opt in ("-o", "--output-type"):
            output_type = arg
        elif opt in ("-t", "--tune"):
            tune = arg
        elif opt in ("-n", "--dry-run"):
            dry_run = True
        elif opt in ("-k", "--keep"):
            keep = True
        elif opt in ("-l", "--prevent-larger"):
            prevent_larger_file = arg != "false"
        elif opt in ("-w", "--hwaccel"):
            if arg != "false":
                hwaccel_requested = arg
        elif opt in ("-s", "--stereo"):
            stereo = True
        elif opt in ("-r", "--rerun"):
            rerun = True
        elif opt == "--no-rerun":
            rerun = False
        elif opt in ("-p", "--preset"):
            preset = arg
        elif opt in ("-f", "--framerate"):
            desired_frame_rate = FRAME_RATE_NAMES.get(arg, arg)
        elif opt in ("-c", "--ignore-errors"):
            ignore_errors = True
        elif opt == "--profanity-filter":
            profanity_filter = True
        elif opt == "--crop-frame":
            crop_frame = True

    if len(args) == 0:
        return None

    options = {
        'desired_video_codecs': desired_video_codecs,
        'desired_audio_codecs': desired_audio_codecs,
        'desired_frame_rate': desired_frame_rate,
        'desired_height': desired_height,
        'preset': preset,
        'prevent_larger_file': prevent_larger_file,
        'output_type': output_type,
        'tune': tune,
        'hwaccel_requested': hwaccel_requested,
        'dry_run': dry_run,
        'keep': keep,
        'stereo': stereo,
        'rerun': rerun,
        'profanity_filter': profanity_filter,
        'crop_frame': crop_frame,
        'ignore_errors': ignore_errors,
    }

    return args, options


def dvr_post_process(*args, **kwargs):
    parsed = parse_args(args)
    if parsed is None:
        return do_dvr_post_process(args, kwargs)
    else:
        merged_args = {**parsed[1], **kwargs}
        return do_dvr_post_process(parsed[0][0], **merged_args)


@common.finisher
def do_dvr_post_process(input_file,
                        # real video codec is resolved per stream and "None" indicates for us to pick the best
                        desired_video_codecs: Iterable[str] = None,
                        desired_audio_codecs: Iterable[str] = None,
                        # NTSC: desired_frame_rate = "30000/1001" or "ntsc", "film" or "24.0", "ntsc_film" or "24000/1001"
                        desired_frame_rate=None,
                        # None keeps the original height, specifying keeps the aspect ratio specified by DAR
                        desired_height=None,
                        # medium is the ffmpeg default
                        preset=None,
                        # If defined, keep the original file if the transcoded file is larger
                        prevent_larger_file=common.get_global_config_boolean('post_process', 'prevent_larger'),
                        output_type="mkv",
                        tune=None,
                        # True to attempt to use hardware acceleration for decoding and encoding as available. Falls back to software.
                        hwaccel_requested=None,
                        dry_run=False,
                        keep=False,
                        stereo=False,
                        rerun=None,
                        profanity_filter=common.get_global_config_boolean('post_process', 'profanity_filter'),
                        crop_frame=False,
                        ignore_errors=None,
                        verbose=False,
                        require_audio=True,
                        ):
    if preset is None:
        preset = os.environ.get('PRESET', common.get_global_config_option('ffmpeg', 'preset'))

    if desired_video_codecs is None:
        desired_video_codecs = common.get_global_config_option('video', 'codecs').split(',')

    if desired_audio_codecs is None:
        desired_audio_codecs = common.get_global_config_option('audio', 'codecs').split(',')

    if not os.path.isfile(input_file):
        logger.error(f"{input_file} does not exist")
        return 255

    filename = os.path.realpath(input_file)
    base_filename = os.path.basename(filename)
    input_type = base_filename.split(".")[-1]
    dir_filename = os.path.dirname(filename)

    # Temporary File Name for transcoding, we want to keep on the same filesystem as the input
    common.TEMPFILENAME = os.path.join(dir_filename,
                                       f".~{'.'.join(base_filename.split('.')[0:-1])}.transcoded.{output_type}")
    # Hides filename from user UI and Dropbox
    common.HIDDEN_FILENAME = os.path.join(dir_filename, f".~{base_filename}")
    output_filename = os.path.join(dir_filename, f"{'.'.join(base_filename.split('.')[0:-1])}.{output_type}")

    if common.assert_not_transcoding(input_file, exit=False) != 0:
        return 255

    os.nice(12)

    #
    # Find transcoder
    #

    # A limited set of codecs appear to be in the Plex Transcoder 1.19.5
    ffmpeg = common.find_ffmpeg()
    vainfo = "/usr/bin/vainfo"

    input_info = common.find_input_info(filename)
    duration = float(input_info['format']['duration'])

    if dry_run:
        logger.info(f"{input_info}")

    # set if any stream is being transcoded rather than copied
    transcoding = False

    #
    # Find desired audio stream(s)
    #
    audio_info_list = common.find_audio_streams(input_info)
    if not audio_info_list and require_audio:
        logger.error(f"{filename}: Could not find desirable audio stream")
        return 255

    #
    # Find desired video stream
    #
    video_info = common.find_video_stream(input_info)
    if not video_info:
        logger.error("Could not find desirable video stream")
        return 255

    # Get video attributes
    aspect_ratio = video_info.get('display_aspect_ratio')
    height = common.get_video_height(video_info)
    if not height:
        logger.error(f"Could not get height info from {filename}: {video_info}")
        return 255
    width = video_info['width']
    if not width:
        logger.error(f"Could not get width info from {filename}: {video_info}")
        return 255
    frame_rate = video_info['avg_frame_rate']
    input_video_codec = common.resolve_video_codec(video_info['codec_name'])
    scale_height = desired_height and desired_height < height
    if scale_height:
        target_height = desired_height
    else:
        target_height = height
    adjust_frame_rate = desired_frame_rate and round(eval(frame_rate), 0) != round(eval(desired_frame_rate), 0)
    target_video_codec = common.resolve_video_codec(desired_video_codecs, target_height, video_info)

    # Find crop frame dimensions
    # cropdetect output (ffmpeg v4, v5) looks like:
    # [Parsed_cropdetect_1 @ 0x7f8ba9010d40] x1:246 x2:1676 y1:9 y2:1079 w:1424 h:1056 x:250 y:18 pts:51298 t:51.298000 crop=1424:1056:250:18
    crop_frame_filter = None
    if crop_frame:
        logger.info("Running crop frame detection")
        crop_frame_rect_histo = {}
        crop_detect_command = [ffmpeg, '-hide_banner', '-skip_frame', 'nointra', '-i', filename,
                               '-vf', f'cropdetect=limit=0.15:reset={math.ceil(eval(frame_rate) * 3)}',
                               '-f', 'null', '/dev/null']
        logger.info(common.array_as_command(crop_detect_command))
        crop_detect_process = subprocess.Popen(crop_detect_command, stderr=subprocess.PIPE, universal_newlines=True)
        crop_detect_regex = r't:([0-9.]+)\s+(crop=[0-9.:]+)\b'
        crop_line_last_t = None
        crop_line_last_filter = None
        for line in crop_detect_process.stderr:
            m = re.search(crop_detect_regex, line)
            if m:
                crop_line_this_t = float(m.group(1))
                crop_line_this_filter = m.group(2)
                if crop_line_last_t is not None:
                    if crop_line_this_filter == crop_line_last_filter:
                        crop_frame_rect_histo[crop_line_this_filter] = crop_frame_rect_histo.get(crop_line_this_filter,
                                                                                                 0) + (
                                                                               crop_line_this_t - crop_line_last_t)
                crop_line_last_t = crop_line_this_t
                crop_line_last_filter = crop_line_this_filter

                # print(f"crop detect: {int(crop_line_this_t * 100 / duration)}%", end="\033[0G")

        # Filter out undesirable frames
        # 1. Only crop if the width is cropped a noticeable amount. Widescreen content intentionally has top and bottom black frames.
        # 2. Crop is over 10% the duration
        logger.debug("crop raw histo %s", crop_frame_rect_histo)
        crop_frame_rect_histo = dict(
            filter(lambda e: (width - common.get_crop_filter_parts(e[0])[0] >= 40) and e[1] > duration / 10,
                   crop_frame_rect_histo.items()))
        logger.debug("crop filtered histo %s", crop_frame_rect_histo)
        if len(crop_frame_rect_histo) > 0:
            crop_frame_rect_histo_list = list(crop_frame_rect_histo.items())
            crop_frame_rect_histo_list.sort(key=lambda e: e[1], reverse=True)
            logger.debug("crop detect histo %s", crop_frame_rect_histo_list)
            crop_frame_filter = crop_frame_rect_histo_list[0][0]
            if width - common.get_crop_filter_parts(crop_frame_filter)[0] < 40:
                crop_frame_filter = None
        logger.info("crop frame filter is %s", crop_frame_filter)

    logger.debug("input video codec = %s, target video codec = %s", input_video_codec, target_video_codec)
    copy_video = not scale_height and not adjust_frame_rate and crop_frame_filter is None and (
            preset == "copy" or (input_video_codec == target_video_codec and not rerun))

    crf, bitrate, qp = common.recommended_video_quality(target_height, target_video_codec)

    # h264_vaapi wasn't a good idea, the bitrate is much higher than software encoding
    # check if either bitrate is too high or vaapi encoding and re-encode
    if copy_video and preset != "copy" and rerun is not False:
        if 'tags' in video_info:
            tags = video_info['tags']
            if 'ENCODER' in tags:
                if 'vaapi' in tags['ENCODER']:
                    logger.info(f"re-encoding {input_video_codec} due to high bitrate vaapi encoder")
                    copy_video = False
        if copy_video:
            # Re-encode if the bit rate looks too high
            if common.K_BIT_RATE in input_info['format']:
                # calculate video bitrate
                video_bitrate = float(input_info['format'][common.K_BIT_RATE])
                for s in input_info['streams']:
                    if s['codec_type'] != 'video':
                        if common.K_BIT_RATE in s:
                            video_bitrate -= int(s[common.K_BIT_RATE])
                        elif 'tags' in s and 'BPS-eng' in s['tags']:
                            video_bitrate -= int(s['tags']['BPS-eng'])
                # adjust video bitrate by 60 fps increments for comparison purposes
                video_bitrate_fps = float(video_bitrate) / (max(1.0, eval(frame_rate) / 60.0))

                bitrate_threshold = bitrate
                # Increase threshold for short videos because there isn't enough content to compress to our expectations
                if duration < SHORT_VIDEO_SECONDS:
                    bitrate_threshold *= 2

                if bitrate and (video_bitrate_fps / 1024) > bitrate_threshold:
                    logger.info(
                        f"re-encoding {input_video_codec} due to high bitrate {video_bitrate_fps / 1024} > {bitrate_threshold}")
                    copy_video = False

    #
    # Construct command
    #

    # Global arguments
    arguments = [ffmpeg]
    if not verbose:
        arguments.extend(["-loglevel", "error"])
    arguments.extend(['-hide_banner', '-y', '-analyzeduration', common.ANALYZE_DURATION,
                      '-probesize', common.PROBE_SIZE])
    if ignore_errors:
        arguments.extend(['-err_detect', 'ignore_err'])

    # extract_closed_captions = input_info['format']['format_name'] == 'mpegts'
    has_text_subtitle_stream = common.has_stream_with_language(input_info,
                                                               common.CODEC_SUBTITLE,
                                                               [common.CODEC_SUBTITLE_ASS, common.CODEC_SUBTITLE_SRT],
                                                               common.LANGUAGE_ENGLISH)
    has_closed_captions = video_info.get('closed_captions', 0) > 0
    extract_closed_captions = has_closed_captions and not has_text_subtitle_stream

    if extract_closed_captions:
        # We need this because shell escaping is hard
        mount_point = find_mount_point(filename)
        if not mount_point or mount_point == '/' or not os.path.isdir(mount_point) or not os.access(mount_point,
                                                                                                    os.W_OK):
            mount_point = dir_filename
        (filename_clean_link_fd, common.FILENAME_CLEAN_LINK) = mkstemp(dir=mount_point, prefix=".~tmplnk.",
                                                                       suffix='.' + output_type)
        if not os.path.isfile(common.FILENAME_CLEAN_LINK):
            (filename_clean_link_fd, common.FILENAME_CLEAN_LINK) = mkstemp(dir=dir_filename, prefix=".~tmplnk.",
                                                                           suffix='.' + output_type)
            if not os.path.isfile(common.FILENAME_CLEAN_LINK):
                logger.error(f"Could not create temp file link in {mount_point} or {dir_filename}")
                return 255
        os.close(filename_clean_link_fd)
        os.remove(common.FILENAME_CLEAN_LINK)
        os.symlink(os.path.realpath(filename), common.FILENAME_CLEAN_LINK)

        # The lavfi filter extracts closed caption subtitles
        arguments.extend(['-f', 'lavfi', '-i', f"movie={common.FILENAME_CLEAN_LINK}[out+subcc]"])
        # The file reference holding the closed captions
        closed_caption_file = 0
        # The file reference holding the input sources aside from closed captions
        streams_file = 1
    else:
        closed_caption_file = -1
        streams_file = 0

    # Configure hwaccel
    # Warning: hwaccel defaults to false because software is more forgiving with corrupted streams
    gpu_present = hwaccel_requested and os.path.isfile('/proc/cpuinfo') and os.path.isdir('/dev/dri')
    hwaccel_inited = False
    if hwaccel_requested == "auto":
        # let ffmpeg figure it out
        arguments.extend(['-hwaccel', 'auto'])
    elif gpu_present and not copy_video and hwaccel_requested == "full":
        if has_hw_codec(ffmpeg, vainfo, input_video_codec):
            arguments.extend(["-hwaccel:v", "vaapi", "-init_hw_device", "vaapi=vaapi:", "-hwaccel_device", "vaapi",
                              "-filter_hw_device", "vaapi"])
            hwaccel_inited = True

    arguments.extend(["-i", filename])

    # Fixes: Too many packets buffered for output stream (can be due to late start subtitle)
    # http://stackoverflow.com/questions/49686244/ddg#50262835
    arguments.extend(['-max_muxing_queue_size', '1024'])

    current_output_stream = 0

    # Audio encoding
    for audio_info in audio_info_list:
        input_audio_codec = audio_info["codec_name"]
        target_audio_codec = common.resolve_audio_codec(desired_audio_codecs, audio_info)
        # Audio encoding doesn't typically need to be re-encoded
        # copy_audio = preset == "copy" or (input_audio_codec == desired_audio_codec and not rerun)
        copy_audio = preset == "copy" or input_audio_codec == target_audio_codec
        audio_input_stream = f"{streams_file}:{audio_info['index']}"
        channels = audio_info['channels']
        arguments.extend(["-map", audio_input_stream])
        if stereo and channels > 2:
            transcoding = True
            copy_audio = False
            arguments.extend([f"-ac:{current_output_stream}", "2"])

        if copy_audio:
            arguments.extend([f"-c:{current_output_stream}", "copy"])
        else:
            transcoding = True
            arguments.extend([f"-c:{current_output_stream}", common.ffmpeg_codec(target_audio_codec)])
            audio_bitrate = int(audio_info[common.K_BIT_RATE]) if common.K_BIT_RATE in audio_info else None
            if target_audio_codec == 'opus':
                common.extend_opus_arguments(arguments, audio_info, current_output_stream, [], stereo)
            elif target_audio_codec == 'aac':
                # use original bit rate if lower than default
                if audio_bitrate is not None:
                    if audio_bitrate < (64 * 1024 * channels):
                        arguments.extend([f"-b:{current_output_stream}", str(max(32, audio_bitrate))])

        current_output_stream += 1

    arguments.extend(["-avoid_negative_ts", "disabled", "-start_at_zero", "-copyts", "-async", "1"])
    if input_type == "mkv":
        arguments.extend(["-max_interleave_delta", "0"])

    # Video encoding
    video_input_stream = f"{streams_file}:{video_info['index']}"
    if copy_video:
        arguments.extend(["-map", video_input_stream, f"-c:{current_output_stream}", "copy"])
    elif hwaccel_inited and hwaccel_requested == "full" and has_hw_codec(ffmpeg, vainfo,
                                                                         target_video_codec) and not scale_height and not crop_frame_filter:
        transcoding = True
        # FIXME: Handle scale_height
        # FIXME: Handle crop_frame_filter
        target_framerate = desired_frame_rate if desired_frame_rate else "30"
        arguments.extend(
            ["-map", video_input_stream, f"-c:{current_output_stream}", "-hwaccel_output_format", "vaapi",
             f"{target_video_codec}_vaapi", "-vf",
             f"fps={target_framerate},deinterlace_vaapi,scale_vaapi=format=nv12", "-qp", str(qp)])
        # -b:{video_input_stream} {bitrate}k
    else:
        transcoding = True
        filter_complex = f"[{video_input_stream}]yadif[0];[0]format=pix_fmts=nv12"
        filter_stage = 1
        if crop_frame_filter is not None:
            filter_complex += f"[{filter_stage}];[{filter_stage}]{crop_frame_filter}"
            filter_stage += 1
        if scale_height:
            filter_complex += f"[{filter_stage}];[{filter_stage}]scale=-2:{desired_height}:flags=bicubic"
            filter_stage += 1
        if adjust_frame_rate:
            # "mi_mode=mci" produces better quality but is single-threaded
            filter_complex += f"[{filter_stage}];[{filter_stage}]minterpolate=fps={desired_frame_rate}:mi_mode=blend"
            filter_stage += 1
        filter_complex += f"[v{current_output_stream}]"

        arguments.extend(
            ["-filter_complex",
             filter_complex,
             "-map", f"[v{current_output_stream}]",
             f"-c:{current_output_stream}", common.ffmpeg_codec(target_video_codec),
             f"-crf:{current_output_stream}", str(crf)])

        if aspect_ratio:
            if crop_frame_filter:
                aspect_ratio_parts = [int(i) for i in aspect_ratio.split(':')]
                crop_parts = common.get_crop_filter_parts(crop_frame_filter)
                aspect_ratio_adjusted_w = aspect_ratio_parts[0] * crop_parts[0] / width
                aspect_ratio_adjusted_h = aspect_ratio_parts[1] * crop_parts[1] / height
                aspect_ratio = f"{math.ceil(aspect_ratio_adjusted_w)}:{math.ceil(aspect_ratio_adjusted_h)}"
            arguments.extend([f"-aspect:{current_output_stream}", aspect_ratio])

        # Do not copy Closed Captions, they will be extracted into a subtitle stream
        if target_video_codec == 'h264' and output_type != 'ts':
            arguments.extend(['-a53cc', '0'])

    current_output_stream += 1

    # Subtitle stream
    # .ts doesn't support subtitle streams
    if output_type != 'ts':
        remove_closed_captions = False
        if extract_closed_captions and streams_file > 0:
            # Closed captions
            arguments.extend(
                ["-map", f"{closed_caption_file}:s?", "-c:s", common.CODEC_SUBTITLE_ASS, "-metadata:s:s:0",
                 f"language={common.LANGUAGE_ENGLISH}"])
            transcoding = True
            remove_closed_captions = True
        else:
            for idx, subtitle in enumerate(
                    common.find_streams_by_codec_and_language(input_info, common.CODEC_SUBTITLE, None,
                                                              common.LANGUAGE_ENGLISH)):
                if subtitle['tags'].get('language') == common.LANGUAGE_ENGLISH:
                    remove_closed_captions = True
                arguments.extend(["-map", f"{streams_file}:{subtitle['index']}"])
                codec_subtitle_switcher = {
                    'mov_text': 'ass',
                    'webvtt': 'ass',
                }
                subtitle_codec = codec_subtitle_switcher.get(subtitle["codec_name"], 'copy')
                if subtitle_codec != 'copy':
                    transcoding = True
                arguments.extend([f"-c:s:{idx}", subtitle_codec])

        if remove_closed_captions and has_closed_captions:
            # Remove closed captions from video stream in favor of separate subtitle stream. Each codec is different.
            # https://stackoverflow.com/questions/48177694/removing-eia-608-closed-captions-from-h-264-without-reencode
            # http://www.ffmpeg.org/ffmpeg-bitstream-filters.html#h264_005fmetadata
            if input_video_codec == 'h264':
                arguments.extend(["-bsf:v", "filter_units=remove_types=6"])
                transcoding = True

    # Metadata
    arguments.extend(
        ["-map_chapters", str(streams_file), "-map_metadata", str(streams_file),
         "-c:t", "copy", "-map", f"{streams_file}:t?"])
    if output_type == 'mov':
        arguments.extend(["-c:d", "copy", "-map", f"{streams_file}:d?"])

    if preset != "copy":
        arguments.extend(["-preset", preset])

    if tune is not None:
        arguments.extend(["-tune", tune])

    arguments.append(common.TEMPFILENAME)

    if not transcoding and input_type == output_type:
        logger.info(f"Stream is already in desired format: {input_file}")
        return 0

    if dry_run:
        logger.info(f"{common.array_as_command(arguments)}")
        return 0

    #
    # Starting Transcoding
    #

    # Check again because there is time between the initial check and when we write to the file
    if common.assert_not_transcoding(input_file, exit=False) != 0:
        return 255
    try:
        Path(common.TEMPFILENAME).touch(mode=0o664, exist_ok=False)
    except FileExistsError:
        return 255

    logger.info(f"Starting transcode of {filename} to {common.TEMPFILENAME}")
    logger.info(f"{common.array_as_command(arguments)}")
    subprocess.run(arguments, check=True)

    if os.stat(common.TEMPFILENAME).st_size == 0:
        logger.error(f"Output at {common.TEMPFILENAME} is zero length")
        return 255

    #
    # Encode Done. Performing Cleanup
    #
    logger.info(f"Finished transcode of {filename} to {common.TEMPFILENAME}")

    filename_stat = os.stat(filename)
    tempfilename_stat = os.stat(common.TEMPFILENAME)

    if prevent_larger_file and filename_stat.st_size < tempfilename_stat.st_size:
        # TODO: need a way to mark as creating a larger file to prevent processing again
        logger.info(f"Original file size is smaller, {filename_stat.st_size} < {tempfilename_stat.st_size}"
                    f", keeping the original")
        return 0

    common.match_owner_and_perm(target_path=common.TEMPFILENAME, source_path=filename)

    # Hide original file in case OUTPUT_TYPE is the same as input
    os.replace(filename, common.HIDDEN_FILENAME)
    try:
        os.replace(common.TEMPFILENAME, output_filename)
    except OSError:
        # Put original file back as fall back
        os.replace(common.HIDDEN_FILENAME, filename)
        logger.error(f"Failed to move converted file: {common.TEMPFILENAME}")
        return 255

    # If we're in the .grab folder, remove original
    grab_folder = ".grab" in os.getcwd() or ".grab" in filename
    if grab_folder or not keep:
        os.remove(common.HIDDEN_FILENAME)

    logger.info("encode done")

    # TODO: It'd be best to run this on the temp file, but we're using a shared common.TEMPFILENAME so it doesn't work
    if profanity_filter:
        if output_type == "mkv":
            do_profanity_filter(output_filename, dry_run=dry_run, verbose=verbose)
        else:
            logger.warning("profanity filter requested but requires mkv, skipping")

    return 0


def dvr_post_process_cli(argv):
    parsed = parse_args(argv)
    if parsed is None:
        usage()
        return 255

    args = parsed[0]
    parsed[1]['verbose'] = True

    atexit.register(common.finish)

    return_code = 0
    for infile, outfile in common.generate_video_files(args, suffix=None):
        # TODO: allow a different outfile
        this_file_return_code = do_dvr_post_process(infile, **parsed[1])
        if this_file_return_code != 0 and return_code == 0:
            return_code = this_file_return_code

    return return_code


if __name__ == '__main__':
    common.setup_cli()
    sys.exit(dvr_post_process_cli(sys.argv[1:]))
