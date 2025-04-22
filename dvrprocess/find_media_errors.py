#!/usr/bin/env python3
import getopt
import logging
import os
import subprocess
import sys
import time
from collections.abc import Iterable

import numpy as np
from scipy.signal import find_peaks
from statsmodels import robust

import common
from common import tools, config

logger = logging.getLogger(__name__)

ERROR_THRESHOLD = 300


def usage():
    print(f"""{sys.argv[0]} [media_paths]

List media files with errors that exceed a threshold.

Output options:

1. Absolute paths terminated with null (this can be changed) with the intent to be piped into xargs or similar tool.
2. Nagios monitoring output, which is also human readable. This also provides some estimates on time to transcode.

--verbose
-t, --terminator="\\n"
    Set the output terminator, defaults to null (0).
-d, --dir=
    Directory containing media. Defaults to {common.get_media_roots()}
--nagios
    Output for Nagios monitoring. Also human readable with statistics and estimates of transcode time.
--cache-only
    Only report cached results, do not look for new media errors.
--time-limit={config.get_global_config_option('background_limits', 'time_limit')}
    Limit runtime. Set to 0 for no limit.
--ignore-compute
    Ignore current compute availability.
""", file=sys.stderr)


def find_media_errors_cli(argv):
    roots = []
    terminator = '\0'
    nagios_output = False
    time_limit = config.get_global_config_time_seconds('background_limits', 'time_limit')
    check_compute = True
    cache_only = False

    try:
        opts, args = getopt.getopt(argv, "t:d:",
                                   ["terminator=", "dir=", "nagios", "time-limit=", "ignore-compute", "cache-only", "verbose"])
    except getopt.GetoptError:
        usage()
        return 2
    for opt, arg in opts:
        if opt == '-h':
            usage()
            return 2
        elif opt in ("-d", "--dir"):
            roots.append(arg)
        elif opt == '--nagios':
            nagios_output = True
        elif opt in ("-t", "--terminator"):
            if arg == '\\n':
                terminator = '\n'
            elif arg == '\\0':
                terminator = '\0'
            else:
                terminator = arg
        elif opt in ['--time-limit']:
            time_limit = config.parse_seconds(arg)
        elif opt == '--ignore-compute':
            check_compute = False
        elif opt == '--cache-only':
            cache_only = True
        elif opt == "--verbose":
            logging.getLogger().setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)

    if not roots:
        roots = common.get_media_roots()

    if args:
        media_paths = common.get_media_paths(roots, args)
    else:
        media_paths = common.get_media_paths(roots)
    logger.debug("media_paths = %s", media_paths)

    if common.check_already_running(quiet=True):
        cache_only = True

    generator = media_errors_generator(media_paths=media_paths, media_roots=roots,
                                       time_limit=time_limit, check_compute=check_compute, cache_only=cache_only)

    if nagios_output:
        corrupt_files = list(generator)
        corrupt_files.sort(key=lambda e: e.error_count, reverse=True)

        if len(corrupt_files) > 25:
            level = "CRITICAL"
            code = 2
        elif len(corrupt_files) > 0:
            level = "WARNING"
            code = 1
        else:
            level = "OK"
            code = 0

        print(f"MEDIA_ERRORS {level}: files: {len(corrupt_files)} | MEDIA_ERRORS;{len(corrupt_files)}")
        for e in corrupt_files:
            print(f"{e.file_name};{e.error_count};eas={str(e.eas_detected).lower()};silence={str(e.silence_detected).lower()}")
        return code
    else:
        for e in generator:
            sys.stdout.write(e.file_name)
            sys.stdout.write(terminator)
        return 0


class MediaErrorFileInfo(object):

    def __init__(self, file_name: str, host_file_path: str, size: float, error_count: int, eas_detected: bool, silence_detected: bool):
        self.file_name = file_name
        self.host_file_path = host_file_path
        self.size = size
        self.error_count = error_count
        self.eas_detected = eas_detected
        self.silence_detected = silence_detected


def detect_eas_tones(filepath) -> bool:
    # Audio stream parameters
    sample_rate = 44100
    dtype = np.int16  # Data type for 16-bit PCM

    ffmpeg_cmd = [
        '-hide_banner',
        '-loglevel', 'error',
        '-nostdin',
        '-i', filepath,
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-ar', str(sample_rate),
        '-ac', '1',
        '-'
    ]
    process = tools.ffmpeg.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    try:
        # Define the window size and overlap for the FFT
        window_duration = 0.3  # Duration of each window in seconds
        window_size = int(sample_rate * window_duration)
        overlap = int(window_size * 0.5)  # 50% overlap

        # Initialize a buffer to hold audio data
        buffer_size = window_size - overlap
        audio_buffer = np.array([], dtype=dtype)

        # Initialize variables for dynamic thresholding
        magnitude_1562_list = []
        magnitude_2083_list = []
        time_stamps = []

        time_position = 0.0

        while True:
            raw_data = process.stdout.read(buffer_size * dtype().nbytes)
            if not raw_data:
                break

            audio_chunk = np.frombuffer(raw_data, dtype=dtype)

            audio_buffer = np.concatenate((audio_buffer, audio_chunk))

            # If the buffer has enough data, process it
            if len(audio_buffer) >= window_size:
                # Extract the current window
                window_data = audio_buffer[:window_size]

                # Apply a Hamming window to reduce spectral leakage
                windowed_data = window_data * np.hamming(len(window_data))

                # Perform FFT
                fft_spectrum = np.fft.rfft(windowed_data)
                freq = np.fft.rfftfreq(len(windowed_data), d=1. / sample_rate)

                # Compute magnitude spectrum in dB
                magnitude = np.abs(fft_spectrum)
                magnitude_db = 20 * np.log10(magnitude + 1e-6)  # Add epsilon to avoid log(0)

                # Detect frequencies near EAS tones
                idx_1562 = np.where((freq >= 1560) & (freq <= 1565))[0]
                idx_2083 = np.where((freq >= 2080) & (freq <= 2085))[0]

                # Calculate average magnitude at target frequencies
                avg_magnitude_1562 = np.mean(magnitude_db[idx_1562]) if idx_1562.size > 0 else -np.inf
                avg_magnitude_2083 = np.mean(magnitude_db[idx_2083]) if idx_2083.size > 0 else -np.inf

                # Append magnitudes and time stamps to lists
                magnitude_1562_list.append(avg_magnitude_1562)
                magnitude_2083_list.append(avg_magnitude_2083)
                time_stamps.append(time_position)

                # Update time position
                time_position += (window_size - overlap) / sample_rate

                # Remove the processed window from the buffer
                audio_buffer = audio_buffer[window_size - overlap:]

        # After processing, convert lists to numpy arrays
        magnitude_1562_array = np.array(magnitude_1562_list)
        magnitude_2083_array = np.array(magnitude_2083_list)
        time_stamps_array = np.array(time_stamps)

        median_1562: float = float(np.median(magnitude_1562_array))
        mad_1562: float = robust.mad(magnitude_1562_array)
        median_2083: float = float(np.median(magnitude_2083_array))
        mad_2083: float = robust.mad(magnitude_2083_array)

        # Set the threshold as median + N * MAD
        N = 3
        threshold_1562: float = median_1562 + N * mad_1562
        threshold_2083: float = median_2083 + N * mad_2083

        # Use the higher threshold for peak detection to reduce false positives
        threshold = max(threshold_1562, threshold_2083)

        # Use peak detection to find peaks in the magnitude arrays
        peaks_1562, _ = find_peaks(magnitude_1562_array, height=threshold)
        peaks_2083, _ = find_peaks(magnitude_2083_array, height=threshold)

        # Find overlapping peaks (peaks that occur at the same time)
        peaks_common = np.intersect1d(peaks_1562, peaks_2083)

        # Report detected EAS tones
        if peaks_common.size >= 2:
            for peak_idx in peaks_common:
                peak_time = time_stamps_array[peak_idx]
                logger.debug(f"{filepath}: EAS tones detected at {common.seconds_to_timespec(peak_time)} seconds.")
            return True

        logger.debug(f"{filepath}: No EAS tones detected.")
        return False

    finally:
        process.terminate()


def detect_silent_audio(filepath) -> bool:
    ffmpeg_cmd = [
        '-hide_banner',
        '-nostdin',
        '-i', filepath,
        '-af', 'silencedetect=noise=-50dB:d=300',
        '-f', 'null',
        '-'
    ]

    output = tools.ffmpeg.check_output(ffmpeg_cmd, stderr=subprocess.STDOUT, text=True)
    silence_lines = list(filter(lambda e: "silencedetect" in e, output.splitlines()))
    if len(silence_lines) > 0:
        logger.debug(f"{filepath}: Silence detected.")
        return True
    else:
        logger.debug(f"{filepath}: No silence detected.")
        return False


def media_errors_generator(media_paths: list[str], media_roots: list[str],
                           time_limit=config.get_global_config_time_seconds('background_limits', 'time_limit'),
                           check_compute=True, cache_only=False) -> Iterable[MediaErrorFileInfo]:
    time_start = time.time()

    for media_path in media_paths:
        for root, dirs, files in os.walk(media_path, topdown=True):
            for file in common.filter_for_mkv(files):
                filepath = os.path.join(root, file)
                if common.is_file_in_hidden_dir(filepath):
                    continue

                cached_error_count = config.get_file_config_option(filepath, 'error', 'count')
                cached_eas_detected = config.get_file_config_option(filepath, 'error', 'eas')
                cached_silence_detected = config.get_file_config_option(filepath, 'error', 'silence')
                if cached_error_count is None or cached_eas_detected is None or cached_silence_detected is None:
                    # We need to calculate one of these, check if we should
                    if cache_only:
                        continue
                    duration = time.time() - time_start
                    if 0 < time_limit < duration:
                        logger.debug(
                            f"Time limit expired after processing {common.s_to_ts(int(duration))}, limit of {common.s_to_ts(time_limit)} reached, only using cached data")
                        cache_only = True
                        continue
                    if check_compute and common.should_stop_processing():
                        # when compute limit is reached, use cached data
                        logger.debug("not enough compute available, only using cached data")
                        cache_only = True
                        continue

                if cached_error_count is not None:
                    error_count = int(cached_error_count)
                else:
                    error_count = len(tools.ffmpeg.check_output(
                        ['-y', '-v', 'error', '-i', filepath, '-c:v', 'vnull', '-c:a', 'anull', '-f', 'null',
                         '/dev/null'],
                        stderr=subprocess.STDOUT, text=True).splitlines())
                    config.set_file_config_option(filepath, 'error', 'count', str(error_count))

                if cached_eas_detected is not None:
                    eas_detected = cached_eas_detected.lower() == "true"
                else:
                    eas_detected = detect_eas_tones(filepath)
                    config.set_file_config_option(filepath, 'error', 'eas', str(eas_detected))

                if cached_silence_detected is not None:
                    silence_detected = cached_silence_detected.lower() == "true"
                else:
                    silence_detected = detect_silent_audio(filepath)
                    config.set_file_config_option(filepath, 'error', 'silence', str(silence_detected))

                if error_count <= ERROR_THRESHOLD and not eas_detected and not silence_detected:
                    continue
                file_info = MediaErrorFileInfo(
                    file_name=common.get_media_file_relative_to_root(filepath, media_roots)[0],
                    host_file_path=filepath,
                    size=os.stat(filepath).st_size,
                    error_count=error_count,
                    eas_detected=eas_detected,
                    silence_detected=silence_detected,
                )
                yield file_info


if __name__ == '__main__':
    os.nice(15)
    common.setup_cli(level=logging.ERROR, start_gauges=False)
    sys.exit(find_media_errors_cli(sys.argv[1:]))
