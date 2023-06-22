#!/usr/bin/env python3
import getopt
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from statistics import mean, stdev
from subprocess import PIPE, DEVNULL

from pysrt import SubRipItem, SubRipFile, SubRipTime
from vosk import Model, KaldiRecognizer, GpuInit, GpuThreadInit

import common
from common import tools, constants, edl_util


def usage():
    print(f"""{sys.argv[0]}
audio transcriber debugging

-w, --words=file_path
    path to store transcribed words in subrip subtitle format
-t, --text=file_path
    path to store transcribed text in subrip subtitle format
-f, --freq=16000
-d, --duration=00:10:00
    limit duration using hh:mm:ss or number of seconds
-b, --buffer=4000
    bytes to process at one time
-a, --audio-filter=
    ffmpeg audio filter, passed to option '-af', example: -a "anlmdn=s=.01"
--max-alt=
    set max alternatives
""", file=sys.stderr)


def audio_transcribe(input_path, freq=16000, words_path=None, text_path=None, buffer_size: int = 4000,
                     duration: [None, str] = None, audio_filter: [None, str] = None,
                     max_alternatives: [None, int] = None):
    """

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
    :return:
    """
    GpuInit()
    GpuThreadInit()

    if audio_filter:
        af_hash = hashlib.sha512(audio_filter.encode('utf-8')).hexdigest()[0:8]
    else:
        af_hash = 'none'
    if duration is None:
        file_prefix = input_path + "-" + af_hash + "-" + str(freq)
    else:
        file_prefix = input_path + "-" + af_hash + "-" + duration + "-" + str(freq) + "-" + str(buffer_size)

    if words_path is None:
        words_path = file_prefix + "-words.srt"
    if text_path is None:
        text_path = file_prefix + ".srt"

    logging.getLogger().setLevel('ERROR')
    model = Model(model_name="vosk-model-en-us-0.22-lgraph", lang="en-us")
    rec = KaldiRecognizer(model, freq)
    rec.SetWords(True)
    if max_alternatives is not None:
        rec.SetMaxAlternatives(max_alternatives)

    input_info = common.find_input_info(input_path)
    ffmpeg_command = ['-nostdin', '-loglevel', 'quiet', '-i', input_path]
    audio_original, audio_filtered, _, _ = common.find_original_and_filtered_streams(input_info, constants.CODEC_AUDIO)
    if not audio_original:
        audio_original = common.find_audio_streams(input_info)[0]
    ffmpeg_command.extend(['-map', f"0:{audio_original.get(constants.K_STREAM_INDEX)}", '-ar', str(freq)])

    if duration is not None:
        ffmpeg_command.extend(['-to', str(edl_util.parse_edl_ts(duration))])

    if audio_filter:
        if 'pan=' not in audio_filter:
            ffmpeg_command.extend(['-ac', '1'])
        ffmpeg_command.extend(['-af', audio_filter])
    else:
        channels = int(audio_original.get(constants.K_CHANNELS, 0))
        if channels > 2:
            ffmpeg_command.extend(['-af', 'pan=1c|FC<0.5*FL+FC+0.5*FR'])
        else:
            ffmpeg_command.extend(['-ac', '1'])

    ffmpeg_command.extend(['-f', 's16le', '-'])
    print(tools.ffmpeg.array_as_command(ffmpeg_command))
    ffmpeg = tools.ffmpeg.Popen(ffmpeg_command, stdout=PIPE, stderr=DEVNULL)
    wf = ffmpeg.stdout
    wf.read(44)  # skip header

    words = []
    while True:
        data = wf.read(buffer_size)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if "text" in result and result["text"] == 'the':
                continue
            if "result" in result:
                words.extend(result["result"])
            print('.', end='', flush=True, file=sys.stderr)

    print('.', flush=True, file=sys.stderr)

    result = json.loads(rec.FinalResult())
    wf.close()
    ffmpeg_return_code = ffmpeg.wait()
    if ffmpeg_return_code != 0:
        return ffmpeg_return_code

    if "text" not in result or result["text"] != 'the':
        if "result" in result:
            words.extend(result["result"])

    confs = []
    for w in words:
        confs.append(w['conf'])
    conf_min = min(confs)
    conf_max = max(confs)
    conf_avg = mean(confs)
    conf_stdev = stdev(confs)

    notes = f'# freq {freq} buf {buffer_size} af {audio_filter} conf [{conf_min},{conf_max}] {conf_avg}σ{conf_stdev}'
    subs_text = []
    subs_words = []

    s = SubRipItem(index=len(subs_words), start=SubRipTime(seconds=0), end=SubRipTime(seconds=1), text=notes)
    subs_words.append(s)
    for word in words:
        s = SubRipItem(index=len(subs_words),
                       start=SubRipTime(seconds=word['start']),
                       end=SubRipTime(seconds=word['end']),
                       text=word['word'])
        subs_words.append(s)

    s = SubRipItem(index=len(subs_text), start=SubRipTime(seconds=0), end=SubRipTime(seconds=1), text=notes)
    subs_text.append(s)
    words_per_line = 7
    for j in range(0, len(words), words_per_line):
        line = words[j: j + words_per_line]
        s = SubRipItem(index=len(subs_text),
                       start=SubRipTime(seconds=line[0]['start']),
                       end=SubRipTime(seconds=line[-1]['end']),
                       text=' '.join([l['word'] for l in line]))
        subs_text.append(s)

    srt_words = SubRipFile(items=subs_words, path=words_path)
    srt_words.save(Path(words_path), 'utf-8')

    srt = SubRipFile(items=subs_text, path=text_path)
    srt.save(Path(text_path), 'utf-8')

    return 0


def audio_transcribe_cli(argv):
    freq = 16000
    words_path = None
    text_path = None
    duration = None
    buffer_size = 4000
    audio_filter = None
    max_alternatives = None

    try:
        opts, args = getopt.getopt(
            list(argv), "w:t:f:d:b:a:",
            ["words=", "text=", "freq=", "duration=", "buffer=", 'audio-filter=', 'max-alt='])
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
        elif opt in ("-d", "--duration"):
            duration = arg
        elif opt in ("-b", "--buffer"):
            buffer_size = int(arg)
        elif opt in ("-a", "--audio-filter"):
            audio_filter = arg
        elif opt == "--max-alt":
            max_alternatives = int(arg)

    if len(args) == 0:
        usage()
        sys.exit(255)

    input_path = args[0]

    audio_transcribe(input_path, words_path=words_path, text_path=text_path, freq=freq, duration=duration,
                     buffer_size=buffer_size, audio_filter=audio_filter, max_alternatives=max_alternatives)
    return 0


if __name__ == '__main__':
    os.nice(12)
    common.setup_cli()
    sys.exit(audio_transcribe_cli(sys.argv[1:]))
