#!/usr/bin/env python3

import atexit
import getopt
import logging
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from tempfile import mkstemp

from ass_parser import CorruptAssLineError

import common
from common import crop_frame, hwaccel, tools, config, constants
from profanity_filter import do_profanity_filter

SHORT_VIDEO_SECONDS = 30

logger = logging.getLogger(__name__)


def usage():
    print(f"""Plex DVR Post Processing Script

Transcode input file to .mkv. Will extract subtitle stream from closed caption.

Hardware transcoding: https://trac.ffmpeg.org/wiki/Hardware/VAAPI
Experience with hardware encoding is poor quality and corruption

A file in the same or parent directories of the input file named '.dvrconfig' can contain command line arguments.
The file closest to the input file will be taken. Comments start with '#'.

{sys.argv[0]} [options] file

-n, --dry-run
    Output command that would be used but do nothing.
-k, --keep
    Keep original file in a backup prefixed by ".~".
-v, --vcodec=h264[,hvec,...]
    The video codec: {config.get_global_config_option('video', 'codecs')} (default), h265, mpeg2.
-a, --acodec=opus[,aac,...]
    The audio codec: {config.get_global_config_option('audio', 'codecs')} (default), aac, ac3, ...
-h, --height=480
    Scale down to this height, maintaining aspect ratio.
-o, --output-type=mkv
    Output type: mkv (default), ts, mp4, ...
-l, --prevent-larger=true,false
    Prevent conversion to a larger file (default is {config.get_global_config_boolean('post_process', 'prevent_larger')}).
-w, --hwaccel=false,auto,full
    Enable hardware acceleration, if available (default is {config.get_global_config_option('ffmpeg', 'hwaccel', 'auto')}).
-s, --stereo
    Scale down audio to stereo.
-p, --preset=copy,medium,fast,veryfast
    Set ffmpeg preset, with a special "copy" for only copying streams.
-t, --tune=[animation,...]
    Set ffmpeg tune value.
--no-rerun
    Prevent re-encoding based on excessive bitrate, extraneous closed captions, etc.
-r, --rerun
    Re-encode streams already in the desired codec.
--profanity-filter
    Include profanity filter in output.
--crop-frame
    Detect and crop surrounding frame. Does not modify widescreen formats that have top and bottom frames.
--crop-frame-ntsc
    Detect and crop surrounding frame to one of the NTSC (and HD) common resolutions.
--crop-frame-pal
    Detect and crop surrounding frame to one of the PAL (and HD) common resolutions.
-f, --framerate={','.join(constants.FRAME_RATE_NAMES.keys())},24,30000/1001,...
    Adjust the frame rate. If the current frame rate is close, i.e. 30000/1001 vs. 30, the option is ignored.
-c, --forgiving
    Ignore errors in the stream, as much as can be done. This may still produce an undesired stream, such as out of sync audio. 
--verbose
    Verbose information about the process
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


def parse_args(argv) -> (list[str], dict):
    desired_video_codecs = None
    desired_audio_codecs = None
    desired_frame_rate = None
    desired_height = None
    preset = None
    prevent_larger_file = config.get_global_config_boolean('post_process', 'prevent_larger')
    output_type = "mkv"
    tune = None
    hwaccel_requested = config.get_global_config_option('ffmpeg', 'hwaccel', fallback=None)
    dry_run = False
    keep = False
    stereo = False
    rerun = None
    forgiving = None
    profanity_filter = config.get_global_config_boolean('post_process', 'profanity_filter')
    crop_frame_op = crop_frame.CropFrameOperation.NONE
    verbose = None

    try:
        opts, args = getopt.getopt(common.get_arguments_from_config(argv, '.dvrconfig') + list(argv),
                                   "v:a:h:o:t:p:f:w:nklsrc",
                                   ["vcodec=", "acodec=", "height=", "output-type=", "tune=", "preset=", "framerate=",
                                    "hwaccel=", "dry-run", "keep",
                                    "prevent-larger=", "stereo", "rerun", "no-rerun", "forgiving",
                                    "profanity-filter", "crop-frame", "crop-frame-ntsc", "crop-frame-pal", "verbose"])
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
            if arg == "false":
                hwaccel_requested = None
            else:
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
            desired_frame_rate = constants.FRAME_RATE_NAMES.get(arg, arg)
        elif opt in ("-c", "--forgiving"):
            forgiving = True
        elif opt == "--profanity-filter":
            profanity_filter = True
        elif opt == "--crop-frame":
            crop_frame_op = crop_frame.CropFrameOperation.DETECT
        elif opt == "--crop-frame-ntsc":
            crop_frame_op = crop_frame.CropFrameOperation.NTSC
        elif opt == "--crop-frame-pal":
            crop_frame_op = crop_frame.CropFrameOperation.PAL
        elif opt == "--verbose":
            verbose = True
            logging.getLogger().setLevel(logging.DEBUG)

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
        'crop_frame_op': crop_frame_op,
        'forgiving': forgiving,
    }
    if verbose is not None:
        options['verbose'] = verbose

    return args, options


def dvr_post_process(*args, **kwargs):
    parsed = parse_args(args)
    try:
        if parsed is None:
            merged_args = {**kwargs}
        else:
            args = parsed[0][0]
            merged_args = {**parsed[1], **kwargs}
        return do_dvr_post_process(args, **merged_args)
    except subprocess.CalledProcessError as e:
        if e.returncode in [-8]:
            logger.warning("Received signal %s, trying with forgiving setting", -e.returncode)
            merged_args['forgiving'] = True
            return do_dvr_post_process(args, **merged_args)
        raise e


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
                        prevent_larger_file=config.get_global_config_boolean('post_process', 'prevent_larger'),
                        output_type="mkv",
                        tune=None,
                        # True to attempt to use hardware acceleration for decoding and encoding as available. Falls back to software.
                        hwaccel_requested=None,
                        dry_run=False,
                        keep=False,
                        stereo=False,
                        rerun=None,
                        profanity_filter=config.get_global_config_boolean('post_process', 'profanity_filter'),
                        crop_frame_op: crop_frame.CropFrameOperation = crop_frame.CropFrameOperation.NONE,
                        forgiving=False,
                        verbose=False,
                        require_audio=True,
                        ):
    if preset is None:
        preset = os.environ.get('PRESET', config.get_global_config_option('ffmpeg', 'preset'))

    if desired_video_codecs is None:
        desired_video_codecs = config.get_global_config_option('video', 'codecs').split(',')

    if desired_audio_codecs is None:
        desired_audio_codecs = config.get_global_config_option('audio', 'codecs').split(',')

    if desired_frame_rate is None:
        desired_frame_rate = config.get_global_config_frame_rate('post_process', 'frame_rate', None)

    if not os.path.isfile(input_file):
        logger.error(f"{input_file} does not exist")
        return 255

    filename = os.path.realpath(input_file)
    base_filename = os.path.basename(filename)
    input_type = base_filename.split(".")[-1]
    dir_filename = os.path.dirname(filename)

    # Temporary File Name for transcoding, we want to keep on the same filesystem as the input
    temp_filename = os.path.join(dir_filename,
                                       f".~{'.'.join(base_filename.split('.')[0:-1])}.transcoded.{output_type}")
    # Hides filename from user UI and Dropbox
    hidden_filename = os.path.join(dir_filename, f".~{base_filename}")
    output_filename = os.path.join(dir_filename, f"{'.'.join(base_filename.split('.')[0:-1])}.{output_type}")

    filename_clean_link = None

    if common.assert_not_transcoding(input_file, exit=False) != 0:
        return 255

    os.nice(12)

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
    width = common.get_video_width(video_info)
    if not width:
        logger.error(f"Could not get width info from {filename}: {video_info}")
        return 255
    frame_rate = common.get_frame_rate(video_info)
    input_video_codec = common.resolve_video_codec(video_info['codec_name'])

    scale_height = desired_height and desired_height < height
    if scale_height:
        target_height = desired_height
    else:
        target_height = height

    target_video_codec = common.resolve_video_codec(desired_video_codecs, target_height, video_info)

    if forgiving:
        crop_frame_filter = None
    else:
        crop_frame_filter = crop_frame.find_crop_frame_filter(crop_frame_op, input_info, frame_rate)

    logger.debug("input video codec = %s, target video codec = %s", input_video_codec, target_video_codec)
    copy_video = not scale_height and crop_frame_filter is None and (
            preset == "copy" or (input_video_codec == target_video_codec and not rerun))

    # If we're re-encoding, we'll be more aggressive in adjusting the frame rate
    if preset == "copy":
        adjust_frame_rate = False
    elif copy_video:
        adjust_frame_rate = common.should_adjust_frame_rate(current_frame_rate=frame_rate,
                                                            desired_frame_rate=desired_frame_rate)
        copy_video = not adjust_frame_rate
    else:
        adjust_frame_rate = common.should_adjust_frame_rate(current_frame_rate=frame_rate,
                                                            desired_frame_rate=desired_frame_rate, tolerance=0.05)

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
            if constants.K_BIT_RATE in input_info['format']:
                # calculate video bitrate
                video_bitrate = float(input_info['format'][constants.K_BIT_RATE])
                for s in input_info['streams']:
                    if s['codec_type'] != 'video':
                        if constants.K_BIT_RATE in s:
                            video_bitrate -= int(s[constants.K_BIT_RATE])
                        elif 'tags' in s and 'BPS-eng' in s['tags']:
                            video_bitrate -= int(s['tags']['BPS-eng'])
                        elif s[constants.K_CODEC_NAME] == 'opus':
                            # opus is VBR and therefore doesn't advertise a bitrate, make assumptions
                            video_bitrate -= (s[constants.K_CHANNELS] * 48000)

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
    arguments = []
    arguments.extend(['-hide_banner', '-y', '-analyzeduration', common.ANALYZE_DURATION,
                      '-probesize', common.PROBE_SIZE])
    if forgiving:
        arguments.extend(['-err_detect', 'ignore_err'])

    has_text_subtitle_stream = common.has_stream_with_language(input_info,
                                                               constants.CODEC_SUBTITLE,
                                                               [constants.CODEC_SUBTITLE_ASS,
                                                                constants.CODEC_SUBTITLE_SRT],
                                                               constants.LANGUAGE_ENGLISH)
    has_closed_captions = video_info.get('closed_captions', 0) > 0
    extract_closed_captions = has_closed_captions and not has_text_subtitle_stream

    if extract_closed_captions:
        # use ccextractor, sometimes lavfi corrupts the subtitles, and ccextractor cleans up the text better
        cc_filename = os.path.join(dir_filename, f".~{base_filename}.cc.ass")
        common.TEMPFILENAMES.append(cc_filename)
        cc_command = ["-out=ass", "-trim", "--norollup", "--nofontcolor", "--notypesetting"]
        # cc_command.append("--sentencecap")  # this will re-capitalize mixed case
        # cc_command.append("--videoedited")
        if not verbose:
            cc_command.append("-quiet")
        cc_command.extend([os.path.realpath(filename), "-o", cc_filename])
        logger.info(f"Extracting closed captions transcode of {filename} to {cc_filename}")
        logger.info(tools.ccextractor.array_as_command(cc_command))
        if not dry_run:
            cc_returncode = tools.ccextractor.run(cc_command, check=False)
            # ccextractor creates additional subtitle files that we don't use
            for cc_root, cc_dirs, cc_files in os.walk(dir_filename):
                for file in cc_files:
                    if file.startswith(f".~{base_filename}.cc.") and file.endswith(".ass") and not file.endswith('.cc.ass'):
                        os.remove(os.path.join(cc_root, file))
                    elif file.startswith(f".~{base_filename}.cc."):
                        common.match_owner_and_perm(target_path=os.path.join(cc_root, file), source_path=filename)
        else:
            cc_returncode = 0

        if dry_run or (cc_returncode == 0 and os.stat(cc_filename).st_size > 0):
            arguments.extend(['-i', cc_filename])
        else:
            logger.warning(f"Output at {cc_filename} is zero length, using ffmpeg lavfi to extract")

            # We need this because shell escaping is hard
            mount_point = find_mount_point(filename)
            if not mount_point or mount_point == '/' or not os.path.isdir(mount_point) or not os.access(mount_point,
                                                                                                        os.W_OK):
                mount_point = dir_filename
            for tempdir in mount_point, tempfile.tempdir:
                try:
                    (filename_clean_link_fd, filename_clean_link) = mkstemp(dir=tempdir, prefix=".~tmplnk.",
                                                                                   suffix='.' + output_type)
                    os.close(filename_clean_link_fd)
                    os.remove(filename_clean_link)
                    os.symlink(os.path.realpath(filename), filename_clean_link)
                except OSError:
                    # Some filesystems don't like the temp names? zfs?
                    pass

            if not filename_clean_link or not os.path.isfile(filename_clean_link):
                logger.error(f"Could not create temp file link in {mount_point} or {dir_filename}")
                return 255

            # The lavfi filter extracts closed caption subtitles
            arguments.extend(['-f', 'lavfi', '-i', f"movie={filename_clean_link}[out+subcc]"])

        # The file reference holding the closed captions
        closed_caption_file = 0
        # The file reference holding the input sources aside from closed captions
        streams_file = 1
    else:
        closed_caption_file = -1
        streams_file = 0

    hwaccel.hwaccel_configure(hwaccel_requested, forgiving=forgiving)
    arguments.extend(hwaccel.hwaccel_threads())
    arguments.extend(
        hwaccel.hwaccel_prologue(input_video_codec=input_video_codec, target_video_codec=target_video_codec))
    arguments.extend(hwaccel.hwaccel_decoding(input_video_codec))
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
        if stereo and channels > 2:
            copy_audio = False

        if copy_audio:
            arguments.extend(["-map", audio_input_stream, f"-c:{current_output_stream}", "copy"])
        else:
            transcoding = True
            audio_bitrate = int(audio_info[constants.K_BIT_RATE]) if constants.K_BIT_RATE in audio_info else None
            if target_audio_codec == 'opus':
                common.map_opus_audio_stream(arguments, audio_info, streams_file, str(current_output_stream), None,
                                             stereo)
            else:
                arguments.extend(["-map", audio_input_stream])
                arguments.extend([f"-c:{current_output_stream}", hwaccel.ffmpeg_sw_codec(target_audio_codec)])
                if stereo and channels > 2:
                    arguments.extend([f"-ac:{current_output_stream}", "2"])
                if target_audio_codec == 'aac':
                    # use original bit rate if lower than default
                    if audio_bitrate is not None:
                        if audio_bitrate < (64 * 1024 * (2 if stereo else channels)):
                            arguments.extend([f"-b:{current_output_stream}", str(max(32, audio_bitrate))])

        current_output_stream += 1

    arguments.extend(["-avoid_negative_ts", "disabled", "-start_at_zero", "-copyts", "-async", "1"])
    if input_type == "mkv":
        arguments.extend(["-max_interleave_delta", "0"])

    # Video encoding
    video_encoder_options_tag_value = []
    video_input_stream = f"{streams_file}:{video_info['index']}"
    if copy_video:
        arguments.extend(["-map", video_input_stream, f"-c:{current_output_stream}", "copy"])
        current_output_stream += 1
    else:
        transcoding = True

        encoding_options, encoding_method = hwaccel.hwaccel_encoding(output_stream=str(current_output_stream),
                                                                     codec=target_video_codec, output_type=output_type,
                                                                     tune=tune, preset=preset, crf=crf, qp=qp,
                                                                     target_bitrate=bitrate)
        video_encoder_options_tag_value.extend(encoding_options)

        filter_stage = 0
        filter_complex = f"[{video_input_stream}]yadif"
        if not forgiving:
            if crop_frame_filter is not None:
                filter_complex += f"[{filter_stage}];[{filter_stage}]{crop_frame_filter}"
                filter_stage += 1
            if scale_height:
                filter_complex += f"[{filter_stage}];[{filter_stage}]scale=-2:{desired_height}:flags=bicubic"
                filter_stage += 1
            if adjust_frame_rate:
                # "mi_mode=mci" produces better quality but is single-threaded
                filter_complex += f"[{filter_stage}];[{filter_stage}]{common.fps_video_filter(desired_frame_rate)}"
                filter_stage += 1
        filter_complex += f"[{filter_stage}];[{filter_stage}]format=nv12"
        filter_stage += 1
        if hwaccel.hwaccel_required_hwupload_filter():
            # must be last
            filter_complex += f"[{filter_stage}];[{filter_stage}]hwupload"
            filter_stage += 1
        filter_complex += f"[v{current_output_stream}]"

        arguments.extend(["-filter_complex", filter_complex, "-map", f"[v{current_output_stream}]"])
        arguments.extend(encoding_options)

        if aspect_ratio:
            if crop_frame_filter:
                aspect_ratio_parts = [int(i) for i in aspect_ratio.split(':')]
                crop_parts = crop_frame.get_crop_filter_parts(crop_frame_filter)
                aspect_ratio_adjusted_w = aspect_ratio_parts[0] * crop_parts[0] / width
                aspect_ratio_adjusted_h = aspect_ratio_parts[1] * crop_parts[1] / height
                aspect_ratio = f"{math.ceil(aspect_ratio_adjusted_w)}:{math.ceil(aspect_ratio_adjusted_h)}"
            arguments.extend([f"-aspect:{current_output_stream}", aspect_ratio])

        current_output_stream += 1

    # Attached pictures
    for attached_pic in common.find_attached_pic_stream(input_info):
        arguments.extend(
            ["-map", f"{streams_file}:{attached_pic[constants.K_STREAM_INDEX]}", f"-c:{current_output_stream}", "copy",
             f"-disposition:{current_output_stream}", "attached_pic"])
        current_output_stream += 1

    # Subtitle stream
    # .ts doesn't support subtitle streams
    if output_type != 'ts':
        remove_closed_captions = False
        if extract_closed_captions and streams_file > 0:
            # Closed captions
            arguments.extend(
                ["-map", f"{closed_caption_file}:s?", "-c:s", constants.CODEC_SUBTITLE_ASS, "-metadata:s:s:0",
                 f"language={constants.LANGUAGE_ENGLISH}"])
            transcoding = True
            remove_closed_captions = True
        else:
            for idx, subtitle in enumerate(
                    common.find_streams_by_codec_and_language(input_info, constants.CODEC_SUBTITLE, None,
                                                              constants.LANGUAGE_ENGLISH)):
                if subtitle['tags'].get('language') == constants.LANGUAGE_ENGLISH:
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
    arguments.extend(['-metadata', f"{constants.K_MEDIA_PROCESSOR}={constants.V_MEDIA_PROCESSOR}"])
    if common.should_replace_media_title(input_info):
        arguments.extend(['-metadata', f"{constants.K_MEDIA_TITLE}={common.get_media_title_from_filename(input_info)}"])
    if len(video_encoder_options_tag_value) > 0:
        arguments.extend(['-metadata', f"{constants.K_ENCODER_OPTIONS}={' '.join(video_encoder_options_tag_value)}"])
    if output_type == 'mov':
        arguments.extend(["-c:d", "copy", "-map", f"{streams_file}:d?"])

    arguments.append(temp_filename)

    if not transcoding and input_type == output_type:
        logger.info(f"Stream is already in desired format: {input_file}")
        return 0

    if dry_run:
        logger.info(f"{tools.ffmpeg.array_as_command(arguments)}")
        return 0

    #
    # Starting Transcoding
    #

    # Check again because there is time between the initial check and when we write to the file
    if common.assert_not_transcoding(input_file, exit=False) != 0:
        return 255
    try:
        Path(temp_filename).touch(mode=0o664, exist_ok=False)
    except FileExistsError:
        return 255

    logger.info(f"Starting transcode of {filename} to {temp_filename}")
    logger.info(f"{tools.ffmpeg.array_as_command(arguments)}")
    tools.ffmpeg.run(arguments, check=True)

    if filename_clean_link and os.path.exists(filename_clean_link):
        try:
            os.remove(filename_clean_link)
        except OSError:
            logger.warning("Could not remove clean link %s", filename_clean_link)

    if os.stat(temp_filename).st_size == 0:
        logger.error(f"Output at {temp_filename} is zero length")
        return 255

    #
    # Encode Done. Performing Cleanup
    #
    logger.info(f"Finished transcode of {filename} to {temp_filename}")

    filename_stat = os.stat(filename)
    tempfilename_stat = os.stat(temp_filename)

    if prevent_larger_file and filename_stat.st_size < tempfilename_stat.st_size:
        # TODO: need a way to mark as creating a larger file to prevent processing again
        logger.info(f"Original file size is smaller, {filename_stat.st_size} < {tempfilename_stat.st_size}"
                    f", keeping the original")
        return 0

    common.match_owner_and_perm(target_path=temp_filename, source_path=filename)

    # Hide original file in case OUTPUT_TYPE is the same as input
    os.replace(filename, hidden_filename)
    try:
        os.replace(temp_filename, output_filename)
    except OSError:
        # Put original file back as fall back
        os.replace(hidden_filename, filename)
        logger.error(f"Failed to move converted file: {temp_filename}")
        return 255

    # If we're in the .grab folder, remove original
    grab_folder = ".grab" in os.getcwd() or ".grab" in filename
    if grab_folder or not keep:
        os.remove(hidden_filename)

    logger.info("encode done")

    # TODO: It'd be best to run this on the temp file, but we're using a shared temp_filename so it doesn't work
    if profanity_filter:
        if output_type == "mkv":
            try:
                do_profanity_filter(output_filename, dry_run=dry_run, verbose=verbose)
            except CorruptAssLineError:
                logger.error("Corrupt ASS subtitle in %s", input_file)
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
    for infile, outfile in common.generate_video_files(args, suffix=None, fail_on_missing=True):
        # TODO: allow a different outfile
        try:
            this_file_return_code = do_dvr_post_process(infile, **parsed[1])
        except subprocess.CalledProcessError as e:
            if not parsed[1].get('forgiving', False) and e.returncode in [-8]:
                logger.warning("Received signal %s, trying with forgiving setting", -e.returncode)
                merged_args = parsed[1].copy()
                merged_args['forgiving'] = True
                this_file_return_code = do_dvr_post_process(infile, **merged_args)
            else:
                raise e
        if this_file_return_code != 0 and return_code == 0:
            return_code = this_file_return_code

    return return_code


if __name__ == '__main__':
    common.cli_wrapper(dvr_post_process_cli)
