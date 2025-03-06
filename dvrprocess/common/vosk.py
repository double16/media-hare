import json
import logging
import os
import weakref
from typing import Union

from vosk import Model, KaldiRecognizer, SetLogLevel
from websockets.sync.client import connect as wsconnect

logger = logging.getLogger(__name__)


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
    See https://alphacephei.com/vosk/models
    :param language: three character language code
    :returns: model name or None to let Vosk choose a model based on the language
    """
    if not language:
        return None
    if language in ['en-us', 'en']:
        # vosk-model-en-us-daanzu-20200905 sometimes is better, but not for the average case
        # return 'vosk-model-en-us-0.22'
        # 'vosk-model-en-us-0.42-gigaspeech' is slightly better than 'vosk-model-en-us-0.22'
        return 'vosk-model-en-us-0.42-gigaspeech'
    elif language == 'es':
        return 'vosk-model-es-0.42'
    return None


class BaseKaldiRecognizer(object):
    def __init__(self, vosk_language: str, freq: int):
        self.vosk_language = vosk_language
        self.freq = freq

    def accept_waveform(self, data) -> int:
        return 0

    def result(self) -> dict:
        return {}

    def final_result(self) -> dict:
        return {}


class LocalKaldiRecognizer(BaseKaldiRecognizer):
    vosk_model_cache = weakref.WeakValueDictionary()

    def __init__(self, vosk_language: str, freq: int, num_channels: int):
        super().__init__(vosk_language, freq)
        SetLogLevel(-99)
        try:
            model = self.vosk_model_cache[self.vosk_language]
        except KeyError:
            model = Model(model_name=vosk_model(self.vosk_language), lang=self.vosk_language)
            self.vosk_model_cache[self.vosk_language] = model
        # self.rec = KaldiRecognizer(model, freq, num_channels)  # num_channels is not an argument
        self.rec = KaldiRecognizer(model, freq)
        self.rec.SetWords(True)

    def accept_waveform(self, data) -> int:
        return self.rec.AcceptWaveform(data)

    def result(self) -> dict:
        return json.loads(self.rec.Result())

    def final_result(self) -> dict:
        try:
            return json.loads(self.rec.FinalResult())
        finally:
            self.rec = None


# map from language to whether server is available
REMOTE_KALDI_SERVER: dict[str, bool] = {}


class RemoteKaldiRecognizer(BaseKaldiRecognizer):
    def __init__(self, vosk_language: str, freq: int, num_channels: int):
        super().__init__(vosk_language, freq)
        lang2 = vosk_language[0:2]
        self.remote_host = os.getenv(f'KALDI_{lang2.upper()}_HOST', f'kaldi-{lang2.lower()}')
        self.remote_port = os.getenv(f'KALDI_{lang2.upper()}_PORT', '2700')
        self.remote_url = f'ws://{self.remote_host}:{self.remote_port}'
        self.socket = wsconnect(self.remote_url)
        self.socket.send(
            '{ "config" : { "words" : true, "max_alternatives" : 0, "sample_rate" : %d, "num_channels": %d } }'
            % (freq, num_channels))

    def accept_waveform(self, data) -> int:
        self.socket.send(data)
        return 1

    def result(self) -> dict:
        message = self.socket.recv()
        if message:
            return json.loads(message)
        else:
            return {}

    def final_result(self) -> dict:
        try:
            self.socket.send('{"eof" : 1}')
            final_message = self.socket.recv()
            if final_message:
                return json.loads(final_message)
            else:
                return {}
        finally:
            self.socket.close()
            self.socket = None


def kaldi_recognizer(language: str, freq: int, num_channels: int) -> BaseKaldiRecognizer:
    global REMOTE_KALDI_SERVER
    _vosk_language = vosk_language(language)
    try:
        if _vosk_language not in REMOTE_KALDI_SERVER or REMOTE_KALDI_SERVER[_vosk_language]:
            rec = RemoteKaldiRecognizer(_vosk_language, freq, num_channels)
            REMOTE_KALDI_SERVER[_vosk_language] = True
            logger.info(f"Using remote transcriber at {rec.remote_url}")
        else:
            rec = LocalKaldiRecognizer(_vosk_language, freq, num_channels)
    except BaseException:
        logger.info("Remote transcriber not available")
        REMOTE_KALDI_SERVER[_vosk_language] = False
        rec = LocalKaldiRecognizer(_vosk_language, freq, num_channels)

    return rec
