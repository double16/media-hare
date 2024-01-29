import sys
from copy import copy
from enum import Enum
from statistics import stdev, StatisticsError
from typing import Union


class EdlType(Enum):
    CUT = 0
    MUTE = 1
    SCENE = 2
    COMMERCIAL = 3
    BACKGROUND_BLUR = 4


class EdlEvent(object):

    def __init__(self, start, end, event_type: EdlType, title: str = None):
        self.start = start
        self.end = end
        self.event_type = event_type
        self.title = title

    def __repr__(self):
        if self.title:
            return f"{self.title}: {self.start} - {self.end}"
        else:
            return f"{self.start} - {self.end}"

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end and self.event_type == other.event_type

    def __copy__(self):
        return EdlEvent(self.start, self.end, self.event_type, self.title)

    def length(self):
        return max(0, self.end - self.start)

    def get_overlap(self, target) -> int:
        if self.start <= target.end and target.start <= self.end:
            return min(self.end, target.end) - max(self.start, target.start)
        else:
            return 0

    def is_overlap_within_percent(self, target, limit_percent: int = 75) -> bool:
        overlap = self.get_overlap(target)
        if overlap <= 0:
            return False
        if self.contains(target) or target.contains(self):
            return True
        min_length = min(self.length(), target.length())
        return (100*(overlap / min_length)) >= limit_percent

    def intersect(self, target):
        if target is None:
            return self
        if self.get_overlap(target) <= 0:
            return None
        result = copy(self)
        result.start = max(result.start, target.start)
        result.end = min(result.end, target.end)
        return result

    def union(self, target):
        if target is None:
            return self
        result = copy(self)
        result.start = min(result.start, target.start)
        result.end = max(result.end, target.end)
        return result

    def contains(self, target):
        if target is None:
            return False
        return target.start >= self.start and target.end <= self.end


def parse_edl(filename) -> list[EdlEvent]:
    events = []
    with open(filename, 'r') as edl_fd:
        for edl_line in edl_fd.readlines():
            if edl_line.startswith('##'):
                continue
            parts = edl_line.replace('-->', ' ').split(maxsplit=3)
            if len(parts) < 3:
                continue
            start = parse_edl_ts(parts[0])
            end = parse_edl_ts(parts[1])
            if parts[2].lower() in ('0', 'cut'):
                event_type = EdlType.CUT
            elif parts[2].lower() in ('1', 'mute'):
                event_type = EdlType.MUTE
            elif parts[2].lower() in ('2', 'scene'):
                event_type = EdlType.SCENE
            elif parts[2].lower() in ('3', 'com', 'commercial'):
                event_type = EdlType.COMMERCIAL
            elif parts[2].lower() in ('4', 'blur'):
                event_type = EdlType.BACKGROUND_BLUR
            else:
                raise Exception(f"Unknown EDL type: {parts[2]}")
            events.append(EdlEvent(start, end, event_type, parts[3].strip() if 3 < len(parts) else None))
    events.sort(key=lambda e: e.start)
    return events


def parse_edl_cuts(filename) -> list[EdlEvent]:
    return list(filter(lambda e: e.event_type in [EdlType.CUT, EdlType.COMMERCIAL], parse_edl(filename)))


def parse_edl_ts(s: str) -> float:
    if ":" in s:
        parts = list(s.split(":"))
        while len(parts) < 3:
            parts.insert(0, "0")
        return round(float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2].replace(',', '.')), 3)
    return round(float(s), 3)


def parse_commercials(filename: str, duration: int = 0) -> (bool, list[EdlEvent], int):
    """
    Find "real" commercials. Removes things too short, or at the beginning or end.
    :param filename: path to EDL file
    :param duration: duration of the entire video, 0 if unknown
    :return: has_com, commercial_breaks, duration_adjustment
    """
    cuts = parse_edl_cuts(filename)
    commercial_breaks = []
    duration_adjustment = 0
    has_com = False
    for idx, event in enumerate(cuts):
        has_com = True
        this_duration = (event.end - event.start)
        duration_adjustment += this_duration
        if event.start == 0:
            # skip portion of recording before show
            pass
        elif duration > 0 and abs(event.end - duration) < 2:
            # skip portion of recording after show
            pass
        elif this_duration >= 20:
            commercial_breaks.append(event)
    return has_com, commercial_breaks, duration_adjustment


def align_commercial_breaks(breaks_in: list[list[EdlEvent]]) -> tuple[list[list[EdlEvent]], Union[float, None], list[EdlEvent]]:
    """
    Align commercial breaks such that each row in the result has the same number of columns. An event may be None.
    Return a score, closer to 0 is better. Maybe None.
    :param breaks_in:
    :return: breaks, score, combined
    """
    result: list[list[EdlEvent]] = []
    if not breaks_in:
        return result, None, []

    # build array of combined breaks
    combined = list()
    for break_in in breaks_in:
        for b in break_in:
            found = False
            for c in combined:
                if b.is_overlap_within_percent(c):
                    found = True
                    c.start = min(c.start, b.start)
                    c.end = max(c.end, b.end)
                    break
            if not found:
                combined.append(copy(b))
    combined.sort(key=lambda e: e.start)
    print("combined:")
    print(pretty_print_commercial_breaks([combined]))

    for break_in in breaks_in:
        resolved = []
        for combined_event in combined:
            resolved_event = None
            for event in break_in:
                if event.is_overlap_within_percent(combined_event):
                    if resolved_event is None:
                        resolved_event = event
                    else:
                        resolved_event = resolved_event.union(event)
            resolved.append(resolved_event)

        result.append(resolved)

    score = 0.001
    for i in range(0, len(combined)):
        try:
            score += stdev(map(lambda e: e.length(), [l[i] for l in result if l[i] is not None]))
        except StatisticsError:
            pass

        score += sum(map(lambda e: 1, [l[i] for l in result if l[i] is None]))

    return result, score, combined


def pretty_print_commercial_breaks(breaks: list[list[EdlEvent]]) -> str:
    def s_to_ts(t: float) -> str:
        hour = int(t / 3600.0)
        minute = int(t / 60) % 60
        second = t % 60.0
        return f"{hour:02d}:{minute:02d}:{second:06.3f}"

    result = ""
    for br in breaks:
        for event in br:
            if event is None:
                result += " "*27 + ", "
            else:
                result += f"{s_to_ts(event.start)} - {s_to_ts(event.end)}, "
        result += "\n"

    return result
