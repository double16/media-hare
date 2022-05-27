#!/usr/bin/env python3

import sys
import re
import pysrt
import hunspell


def words_in_dictionary_pct(subtitle_srt_filename, language):
    # verify with spell checker (hunspell) that text looks like English
    if language == 'eng':
        hobj = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
    else:
        hobj = hunspell.HunSpell(f'/usr/share/hunspell/{language[0:2]}.dic', f'/usr/share/hunspell/{language[0:2]}.aff')
    word_count = 0
    word_found_count = 0
    missing = set()
    srt_data = pysrt.open(subtitle_srt_filename)
    for event in list(srt_data):
        for word in re.sub('[^A-Za-z\' ]+', ' ', event.text).split():
            word_count += 1
            if hobj.spell(word):
                word_found_count += 1
            else:
                missing.add(word)
    word_found_pct = 100.0 * float(word_found_count) / float(word_count)
    print(f"{subtitle_srt_filename}: SRT words = {word_count}, found = {word_found_count}, {word_found_pct}%")
    return word_found_pct


language = 'eng'
word_found_pct = words_in_dictionary_pct(sys.argv[1], language)
if word_found_pct < 93.0:
    print("FAIL")
    sys.exit(2)
else:
    print("PASS")
    sys.exit(0)
