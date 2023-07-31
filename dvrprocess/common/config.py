import configparser
import logging
import os
import sys
import tempfile
import threading
from enum import Enum

from . import constants

_config_lock = threading.Lock()

logger = logging.getLogger(__name__)

KILOBYTES_MULT = 1024
MEGABYTES_MULT = 1024 * 1024
GIGABYTES_MULT = 1024 * 1024 * 1024


def _get_config_sources(filename: str):
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    # TODO: check Windows places
    return [f"{os.environ['HOME']}/.{filename}",
            f"/etc/{filename}",
            f"{script_dir}/{filename}",
            f"./{filename}"]


def find_config(filename: str) -> str:
    """
    Locate a config file from a list of common locations. The first one found is returned.
    See _get_config_sources for locations.
    :param filename: filename, like 'config.ini'
    :return: resolved path to a file that exists
    """
    sources = _get_config_sources(filename)
    for f in sources:
        if os.access(f, os.R_OK):
            return f
    raise OSError(f"Cannot find {filename} in any of {','.join(sources)}")


def load_config(filename: str, start_path=None, input_info=None,
                config: configparser.ConfigParser = None) -> configparser.ConfigParser:
    """
    Load configuration from one or more INI files. Overrides to the common configuration can be made by creating file(s)
    named :filename: in the directory tree of :start_path: or location of :input_info:.

    :param filename: the name of the config file, usually something like 'config.ini'
    :param start_path: the lowest directory or file of the directory tree to search for overrides
    :param input_info: if specified and the info contains the filename, set start_path from this
    :param config: an existing config to update
    :return: ConfigParser object
    """
    filenames = [find_config(filename)]

    if not start_path and input_info:
        input_path = input_info.get(constants.K_FORMAT, {}).get("filename")
        if input_path:
            if os.path.isfile(input_path):
                start_path = os.path.dirname(input_path)
            elif os.path.isdir(input_path):
                start_path = input_path

    if start_path:
        path = start_path
        insert_idx = len(filenames)
        while True:
            p = os.path.dirname(os.path.abspath(path))
            if not p or p == path:
                break
            conf = os.path.join(p, filename)
            if os.path.isfile(conf):
                filenames.insert(insert_idx, conf)
            path = p

    if not config:
        config = configparser.ConfigParser()
    config.read(filenames)

    return config


config_obj: [configparser.ConfigParser, None] = None
_UNSET = object()


def get_global_config_option(section: str, option: str, fallback: [None, str] = _UNSET):
    """
    Get an option from the global config (i.e., media-hare.ini and media-hare.defaults.ini)
    :param section:
    :param option:
    :param fallback:
    :return:
    """
    if fallback is not _UNSET:
        return get_global_config().get(section, option, fallback=fallback)
    return get_global_config().get(section, option)


def get_global_config_boolean(section: str, option: str, fallback: bool = None):
    """
    Get a boolean from the global config (i.e., media-hare.ini and media-hare.defaults.ini)
    :param section:
    :param option:
    :param fallback:
    :return:
    """
    if fallback is not None:
        return get_global_config().getboolean(section, option, fallback=fallback)
    return get_global_config().getboolean(section, option)


def get_global_config_int(section: str, option: str, fallback: int = None):
    """
    Get an int from the global config (i.e., media-hare.ini and media-hare.defaults.ini)
    :param section:
    :param option:
    :param fallback:
    :return:
    """
    if fallback is not None:
        return get_global_config().getint(section, option, fallback=fallback)
    return get_global_config().getint(section, option)


def get_global_config_time_seconds(section: str, option: str, fallback: int = None):
    """
    Get number of seconds from the global config (i.e., media-hare.ini and media-hare.defaults.ini)
    :param section:
    :param option:
    :param fallback:
    :return:
    """
    if fallback is not None:
        return parse_seconds(get_global_config().get(section, option, fallback=fallback))
    return parse_seconds(get_global_config().get(section, option))


def get_global_config_bytes(section: str, option: str, fallback: int = None):
    """
    Get number of bytes from the global config (i.e., media-hare.ini and media-hare.defaults.ini)
    :param section:
    :param option:
    :param fallback:
    :return:
    """
    if fallback is not None:
        return parse_bytes(get_global_config().get(section, option, fallback=fallback))
    return parse_bytes(get_global_config().get(section, option))


def get_work_dir() -> str:
    work_dir = get_global_config_option('general', 'work_dir', fallback=tempfile.gettempdir())
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    return work_dir


def get_global_config_frame_rate(section: str, option: str, fallback: [None, str] = _UNSET) -> [None, str]:
    """
    Get the frame rate from the global config (i.e., media-hare.ini and media-hare.defaults.ini)
    :param section:
    :param option:
    :param fallback:
    :return: numeric frame rate, named frame rates are converted to numeric
    """

    if fallback == _UNSET:
        value = get_global_config().get(section, option)
    else:
        value = get_global_config().get(section, option, fallback=fallback)
    if value is None:
        return None
    return constants.FRAME_RATE_NAMES.get(value.lower(), value)


class MuteChannels(Enum):
    ALL = 0
    VOICE = 1


def get_global_config_mute_channels() -> MuteChannels:
    """
    Get the mute channels config.
    :return: numeric frame rate, named frame rates are converted to numeric
    """

    return mute_channels(get_global_config().get('profanity_filter', 'mute_channels', fallback='all').lower())


def mute_channels(value: str) -> MuteChannels:
    if value == 'voice':
        return MuteChannels.VOICE
    elif value == 'all':
        return MuteChannels.ALL
    else:
        raise configparser.Error(f"mute_channels value unknown: {value}")


def get_global_config() -> configparser.ConfigParser:
    global config_obj
    if config_obj is None:
        _config_lock.acquire()
        try:
            if config_obj is None:
                config_obj = _load_media_hare_config()
        finally:
            _config_lock.release()
    return config_obj


def _load_media_hare_config() -> configparser.ConfigParser:
    defaults = load_config('media-hare.defaults.ini')
    return load_config('media-hare.ini', config=defaults)


def parse_seconds(s: str) -> int:
    if s is None:
        return 0
    s = s.strip().lower()
    if len(s) == 0:
        return 0
    if s.endswith("s"):
        multiplier = 1
    elif s.endswith("m"):
        multiplier = 60
    elif s.endswith("h"):
        multiplier = 3600
    else:
        return int(s)
    return int(float(s[0:-1]) * multiplier)


def parse_bytes(s: str) -> int:
    if s is None:
        return 0
    s = s.strip().lower()
    if len(s) == 0:
        return 0
    if s.endswith("b"):
        multiplier = 1
    elif s.endswith("k"):
        multiplier = KILOBYTES_MULT
    elif s.endswith("m"):
        multiplier = MEGABYTES_MULT
    elif s.endswith("g"):
        multiplier = GIGABYTES_MULT
    else:
        return int(s)
    return int(float(s[0:-1]) * multiplier)


def bytes_to_human_str(byte_count: int) -> str:
    if byte_count > GIGABYTES_MULT:
        return "{:.2f}G".format(float(byte_count) / GIGABYTES_MULT)
    if byte_count > MEGABYTES_MULT:
        return "{:.2f}M".format(float(byte_count) / MEGABYTES_MULT)
    if byte_count > KILOBYTES_MULT:
        return "{:.2f}M".format(float(byte_count) / KILOBYTES_MULT)
    return str(byte_count)
