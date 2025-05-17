#!/usr/bin/env python3
import getopt
import hashlib
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Union

import pysrt
import whisper
from pysrt import SubRipItem, SubRipFile, SubRipTime
from thefuzz import fuzz

import common
from common import tools, constants, edl_util, progress
from dvrprocess.common.proc_invoker import StreamCapturingProgress
from profanity_filter import srt_words_to_sentences, SUBTITLE_TEXT_TO_PLAIN_WS, \
    SUBTITLE_TEXT_TO_PLAIN_SQUEEZE_WS

DEFAULT_MODEL = "medium"
DEFAULT_FREQ = 16000
DEFAULT_BUFFER_SIZE = 4000
DEFAULT_NUM_CHANNELS = 1

logger = logging.getLogger(__name__)


def usage():
    print(f"""{sys.argv[0]}
audio transcriber debugging

-w, --words=file_path
    path to store transcribed words in subrip subtitle format
-t, --text=file_path
    path to store transcribed text in subrip subtitle format
-f, --freq={DEFAULT_FREQ}
-d, --duration=00:10:00
    limit duration using hh:mm:ss or number of seconds
-b, --buffer={DEFAULT_BUFFER_SIZE}
    bytes to process at one time
-c, --channels={DEFAULT_NUM_CHANNELS}
    number of channels, 1 (mono) or 2 (stereo)
-a, --audio-filter=
    ffmpeg audio filter, passed to option '-af', example: -a "anlmdn=s=.01"
--audio-filter-file=
    audio filters, one per line
--max-alt=2
    set max alternatives
""", file=sys.stderr)


WHISPER_MODELS = dict()

def audio_transcribe(input_path, freq=DEFAULT_FREQ, words_path=None, text_path=None,
                     buffer_size: int = DEFAULT_BUFFER_SIZE,
                     num_channels: int = DEFAULT_NUM_CHANNELS,
                     duration: Union[None, str] = None, audio_filter: Union[None, str] = None,
                     no_clobber=True, input_info=None,
                     model_name: Union[str, None] = None):
    """

    :param input_info:
    :param model_name:
    :param input_path:
    :param freq:
    :param words_path:
    :param text_path:
    :param buffer_size:
    :param duration:
    :param audio_filter:
        https://superuser.com/questions/733061/reduce-background-noise-and-optimize-the-speech-from-an-audio-clip-using-ffmpeg
        'lowpass=f=3000,highpass=f=200,afftdn=nf=-25'
        'afftdn=nf=-25'  # https://ffmpeg.org/ffmpeg-filters.html#afftdn
        'anlmdn'  # https://ffmpeg.org/ffmpeg-filters.html#anlmdn
    :param no_clobber: True to not run if the output file(s) already exist
    :return:
    """

    model_name = model_name or DEFAULT_MODEL
    if model_name in WHISPER_MODELS:
        model = WHISPER_MODELS[model_name]
    else:
        model = whisper.load_model(model_name)
        WHISPER_MODELS[model_name] = model


    words_path, text_path = compute_file_paths(
        input_path=input_path,
        audio_filter=audio_filter,
        model_name=model_name,
        duration=duration,
        words_path=words_path,
        text_path=text_path,
        freq=freq,
        num_channels=num_channels,
        buffer_size=buffer_size,
    )

    if no_clobber and os.path.exists(words_path) and os.path.exists(text_path):
        return 0

    if not input_info:
        input_info = common.find_input_info(input_path)

    ffmpeg_command = ['-nostdin', '-loglevel', 'quiet', '-i', input_path]
    audio_original, audio_filtered, _, _ = common.find_original_and_filtered_streams(input_info, constants.CODEC_AUDIO)
    if not audio_original:
        audio_original = common.find_audio_streams(input_info)[0]
    ffmpeg_command.extend(['-map', f"0:{audio_original.get(constants.K_STREAM_INDEX)}", '-ar', str(freq)])

    if duration is not None:
        ffmpeg_command.extend(['-to', str(edl_util.parse_edl_ts(duration))])

    if audio_filter and 'pan=' in audio_filter:
        ffmpeg_command.extend(['-af', audio_filter])
    else:
        channels = int(audio_original.get(constants.K_CHANNELS, 0))
        if channels > 2:
            if num_channels == 1:
                pan='pan=mono|FC<FC+0.5*FL+0.5*FR'
            else:
                pan='pan=stereo|FL<FL+FC|FR<FR+FC'
            if audio_filter:
                ffmpeg_command.extend(['-af', pan+","+audio_filter])
            else:
                ffmpeg_command.extend(['-af', pan])
        else:
            ffmpeg_command.extend(['-ac', str(num_channels)])
            if audio_filter:
                ffmpeg_command.extend(['-af', audio_filter])

    ffmpeg_command.extend(['-f', 'wav'])
    audio_filename = os.path.splitext(words_path)[0] + ".wav"
    ffmpeg_command.extend(['-f', 'wav', audio_filename])

    print(tools.ffmpeg.array_as_command(ffmpeg_command))
    tools.ffmpeg.run(ffmpeg_command, check=True)

    audio_progress = StreamCapturingProgress(
        "stderr",
        progress.progress(
            f"{os.path.basename(input_info['format']['filename'])} transcription",
            0, 100))

    whisper_result = model.transcribe(
        audio=audio_filename,
        language='en',
        fp16=False,
        word_timestamps=True,
        verbose=False,
        beam_size=5,
        patience=2.5
    )

    audio_progress.finish()

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

    if confs:
        conf_min = min(confs)
        conf_max = max(confs)
        conf_avg = mean(confs)
        conf_stdev = stdev(confs)
    else:
        conf_min = None
        conf_max = None
        conf_avg = None
        conf_stdev = None

    notes = f'# model {model_name} freq {freq} buf {buffer_size} af {audio_filter} conf [{conf_min},{conf_max}] {conf_avg}σ{conf_stdev}'
    notes_item = SubRipItem(index=0, start=SubRipTime(seconds=0), end=SubRipTime(seconds=1), text=notes)

    subs_text = srt_words_to_sentences(subs_words, 'en')

    subs_words.insert(0, notes_item)
    srt_words = SubRipFile(items=subs_words, path=words_path)
    srt_words.save(Path(words_path), 'utf-8')

    subs_text.insert(0, notes_item)
    srt = SubRipFile(items=subs_text, path=text_path)
    srt.save(Path(text_path), 'utf-8')

    return 0


def audio_transcribe_cli(argv):
    freq = DEFAULT_FREQ
    words_path = None
    text_path = None
    duration = None
    buffer_size = DEFAULT_BUFFER_SIZE
    num_channels = DEFAULT_NUM_CHANNELS
    audio_filter = None
    audio_filter_file = None

    try:
        opts, args = getopt.getopt(
            list(argv), "w:t:f:d:b:a:c:",
            ["words=", "text=", "freq=", "duration=", "buffer=", 'audio-filter=', "audio-filter-file=",
             "channels="])
    except getopt.GetoptError:
        usage()
        sys.exit(255)
    for opt, arg in opts:
        if opt == '--help':
            usage()
            sys.exit(255)
        elif opt in ("-w", "--words"):
            words_path = arg
        elif opt in ("-t", "--text"):
            text_path = arg
        elif opt in ("-f", "--freq"):
            freq = int(arg)
        elif opt in ("-c", "--channels"):
            num_channels = int(arg)
        elif opt in ("-d", "--duration"):
            duration = arg
        elif opt in ("-b", "--buffer"):
            buffer_size = int(arg)
        elif opt in ("-a", "--audio-filter"):
            audio_filter = arg
        elif opt == "--audio-filter-file":
            audio_filter_file = arg

    if len(args) == 0:
        usage()
        sys.exit(255)

    input_path = args[0]

    if not audio_filter_file:
        audio_transcribe(input_path, words_path=words_path, text_path=text_path, freq=freq, duration=duration,
                         num_channels=num_channels,
                         buffer_size=buffer_size, audio_filter=audio_filter,
                         no_clobber=False)
        return 0

    audio_filter_list = [
        (None, "large-v2"),
    ]
    with open(audio_filter_file, "rt") as audio_filter_fh:
        for line in audio_filter_fh.readlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if '","' in line:
                audio_filter, model_name = tuple(map(lambda e: e.strip('"').strip(), line.split('","')))
            else:
                audio_filter = line.strip()
                model_name = None
            audio_filter_list.append((audio_filter, model_name))
    logger.info("audio filters: %s", audio_filter_list)

    input_info = common.find_input_info(input_path)
    audio_filter_list_progress = progress.progress("audio filters", 1, len(audio_filter_list))
    for af_idx, (audio_filter, model_name) in enumerate(audio_filter_list):
        model_name = model_name or DEFAULT_MODEL
        logger.info("transcribing: audio_filter=%s, model_name=%s", audio_filter, model_name)
        audio_transcribe(input_path, input_info=input_info, freq=freq, duration=duration,
                         buffer_size=buffer_size, audio_filter=audio_filter, model_name=model_name,
                         no_clobber=True)
        audio_filter_list_progress.progress(af_idx + 1, msg=str(audio_filter))
    audio_filter_list_progress.stop()

    if not duration:
        duration_f = float(input_info[constants.K_FORMAT][constants.K_DURATION])
    else:
        duration_f = edl_util.parse_edl_ts(duration)

    expected_text = read_text_from_mkv(input_info, duration_f)
    r = fuzz.WRatio(expected_text, expected_text)
    print(f"{r}: sanity check expected vs. expected")
    for af_idx, (audio_filter, model_name) in enumerate(audio_filter_list):
        model_name = model_name or DEFAULT_MODEL
        _, text_path = compute_file_paths(
            input_path=input_path,
            audio_filter=audio_filter,
            model_name=model_name,
            duration=duration,
            freq=freq,
            buffer_size=buffer_size,
        )
        transcribed_text = read_text_from_srt(text_path, duration_f)
        r = fuzz.WRatio(expected_text, transcribed_text)
        print(f"{r}: {audio_filter},{model_name} - {text_path}")

    return 0


def compute_file_paths(
        input_path: str,
        audio_filter: Union[None, str] = None,
        model_name: Union[None, str] = None,
        duration: Union[None, str] = None,
        words_path: Union[None, str] = None,
        text_path: Union[None, str] = None,
        freq: int = DEFAULT_FREQ,
        num_channels: int = DEFAULT_NUM_CHANNELS,
        buffer_size: int = DEFAULT_BUFFER_SIZE) -> tuple[str, str]:
    """
    :return: words_path, text_path
    """
    hash_input = (audio_filter or 'none') + (model_name or 'none')
    af_hash = hashlib.sha512(hash_input.encode('utf-8')).hexdigest()[0:8]
    if duration is None:
        file_prefix = input_path + "-" + af_hash + "-" + str(freq) + "-" + str(num_channels)
    else:
        file_prefix = input_path + "-" + af_hash + "-" + duration + "-" + str(freq) + "-" + str(
            num_channels) + "-" + str(buffer_size)

    if words_path is None:
        words_path = file_prefix + "-words.srt"
    if text_path is None:
        text_path = file_prefix + ".srt"

    return words_path, text_path


CLEAN_SRT_TEXT = re.compile(r"<.*?>|{.*?}|\[.*?]")


def clean_srt_text(event_text: str) -> str:
    return SUBTITLE_TEXT_TO_PLAIN_SQUEEZE_WS.sub(
        " ",SUBTITLE_TEXT_TO_PLAIN_WS.sub(
            " ", CLEAN_SRT_TEXT.sub(
                "", event_text)))


def read_text_from_srt(path, duration: float) -> str:
    srt = pysrt.open(path)
    text = ' '.join(map(lambda e: clean_srt_text(e.text), filter(lambda e: e.start.ordinal / 1000 < duration, srt)))
    return text


def read_text_from_mkv(input_info: dict, duration: float) -> str:
    subtitles = common.find_streams_by_codec_and_language(input_info, constants.CODEC_SUBTITLE,
                                                          constants.CODEC_SUBTITLE_TEXT_BASED,
                                                          constants.LANGUAGE_ENGLISH)
    if not subtitles:
        raise FileNotFoundError(f"No subtitle in input stream {input_info[constants.K_FORMAT]['filename']}")
    subtitle = list(filter(lambda e: e.get('tags', {}).get(constants.K_STREAM_TITLE) == constants.TITLE_ORIGINAL, subtitles))
    if not subtitle:
        subtitle = subtitles
    args = ['-nostdin', '-loglevel', 'error',
            '-i', input_info['format']['filename']]
    if duration is not None:
        args.extend(['-to', str(duration + 1)])
    args.extend([
        '-map', f'0:{subtitle[0][constants.K_STREAM_INDEX]}',
        '-c', 'srt',
        '-f', 'srt', '-'
    ])
    logger.info('ffmpeg %s', ' '.join(args))
    proc = tools.ffmpeg.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1, text=True)
    srt = pysrt.stream(proc.stdout)
    text = ' '.join(map(lambda e: clean_srt_text(e.text), filter(lambda e: e.start.ordinal / 1000 < duration, srt)))
    proc.stdout.close()
    proc.wait()

    return text


if __name__ == '__main__':
    os.nice(12)
    common.setup_cli()
    sys.exit(audio_transcribe_cli(sys.argv[1:]))
