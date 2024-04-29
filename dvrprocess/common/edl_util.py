import logging
from copy import copy
from enum import Enum
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)


class EdlType(Enum):
    CUT = 0
    MUTE = 1
    SCENE = 2
    COMMERCIAL = 3
    BACKGROUND_BLUR = 4


class EdlEvent(object):

    def __init__(self, start: float, end: float, event_type: EdlType, title: str = None):
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
        if other is None:
            return False
        return self.start == other.start and self.end == other.end and self.event_type == other.event_type

    def __copy__(self):
        return EdlEvent(self.start, self.end, self.event_type, self.title)

    def length(self):
        return max(0, self.end - self.start)

    def get_overlap(self, target) -> int:
        overlap_start = max(self.start, target.start)
        overlap_end = min(self.end, target.end)
        return max(0, overlap_end - overlap_start)

    def is_overlap_within_percent(self, target, limit_percent: int = 40) -> bool:
        overlap = self.get_overlap(target)
        if overlap <= 0:
            return False
        if self.contains(target) or target.contains(self):
            return True
        min_length = min(self.length(), target.length())
        return (overlap / min_length) * 100 >= limit_percent

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


def parse_commercials(filename: str, duration: int = 0, adjust_start_time: bool = False, min_commercial_length: int = 20) -> (bool, list[EdlEvent], int):
    """
    Find "real" commercials. Removes things too short, or at the beginning or end.
    :param filename: path to EDL file
    :param duration: duration of the entire video, 0 if unknown
    :return: has_com, commercial_breaks, duration_adjustment
    """
    cuts = parse_edl_cuts(filename)
    commercial_breaks = []
    duration_adjustment = 0
    start_time_adjustment = 0
    has_com = False
    for idx, event in enumerate(cuts):
        has_com = True
        this_duration = (event.end - event.start)
        duration_adjustment += this_duration
        if event.start == 0 and event.length() < 35:
            # skip portion of recording before show
            if adjust_start_time:
                start_time_adjustment = event.end
        elif duration > 0 and abs(event.end - duration) < 8 and event.length() < 35:
            # skip portion of recording after show
            pass
        elif this_duration >= min_commercial_length:
            event.start -= start_time_adjustment
            event.end -= start_time_adjustment
            commercial_breaks.append(event)
    return has_com, commercial_breaks, duration_adjustment


class CombinedEdlEvents(object):
    def __init__(self, event: EdlEvent):
        self._events: list[EdlEvent] = [event]
        self._reduced: Union[EdlEvent, None] = None

    def get_events(self):
        return self._events

    def is_overlap_within_percent(self, target: EdlEvent) -> bool:
        return any(map(lambda e: e.is_overlap_within_percent(target), self._events))

    def append(self, e: EdlEvent):
        self._events.append(e)
        self._reduced = None

    def append_all(self, events: list[EdlEvent]):
        for e in events:
            self._events.append(e)
        self._reduced = None

    def _average_without_extremes(self, numbers: list[int]):
        if len(numbers) < 5:
            return sum(numbers) / len(numbers)
        return sum(sorted(numbers)[1:-1]) / max(1, len(numbers) - 2)

    def reduce(self) -> EdlEvent:
        if self._reduced is None:
            self._reduced = EdlEvent(
                self._average_without_extremes(list(map(lambda e: e.start, self._events))),
                self._average_without_extremes(list(map(lambda e: e.end, self._events))),
                self._events[0].event_type)
        return self._reduced


def align_commercial_breaks(breaks_in: list[list[EdlEvent]]) -> tuple[
    list[list[EdlEvent]], Union[float, None], list[EdlEvent]]:
    """
    Align commercial breaks such that each row in the result has the same number of columns. An event may be None.
    Return a score, closer to 0 is better. Maybe None.
    :param breaks_in:
    :return: breaks, score, combined
    """
    result: list[list[Union[EdlEvent, None]]] = []
    if not breaks_in:
        return result, None, []

    breaks_in_sorted = sorted(breaks_in, key=lambda b: b[0].start if b and b[0] else 0)

    # build array of combined breaks
    combined: list[CombinedEdlEvents] = list()
    for break_in in breaks_in_sorted:
        for b in break_in:
            overlaps: list[tuple[int, CombinedEdlEvents]] = []
            for c in combined:
                if c.is_overlap_within_percent(b):
                    overlaps.append((b.get_overlap(c.reduce()), c))
            if not overlaps:
                combined.append(CombinedEdlEvents(b))
            else:
                overlaps.sort(key=lambda e: e[0])
                overlaps[-1][1].append(b)
    combined.sort(key=lambda e: e.reduce().start)
    for c in range(len(combined) - 1, 0, -1):
        if combined[c].reduce().get_overlap(combined[c - 1].reduce()) > 0:
            combined[c - 1].append_all(combined.pop(c).get_events())

    for break_in in breaks_in_sorted:
        resolved = []
        break_in_todo = break_in.copy()
        for combined_event in combined:
            resolved_event = None
            consumed_events = []
            for event in break_in_todo:
                if event.is_overlap_within_percent(combined_event.reduce()):
                    consumed_events.append(event)
                    if resolved_event is None:
                        resolved_event = event
                    else:
                        resolved_event = resolved_event.union(event)
            resolved.append(resolved_event)
            for event in consumed_events:
                break_in_todo.remove(event)

        result.append(resolved)

    if (len(combined) * len(result)) == 0:
        score = 0.001
    else:
        # convert to matrices and compute the overall standard deviation
        missing_value = -1
        empty_count = 0
        matrices = list()
        for result_el in result:
            matrices.append(np.array(
                [[missing_value if x is None else x.start, missing_value if x is None else x.end] for x in result_el]))
            empty_count += result_el.count(None)
        stacked_matrices = np.stack(matrices, axis=-1)
        score = 0.001 + np.mean(np.std(stacked_matrices, axis=-1, dtype=np.float64))

        element_count = len(combined) * len(result)
        sparsity_ratio = empty_count / max(1, element_count)
        score += 2000 * sparsity_ratio
    return result, score, list(map(lambda c: c.reduce(), combined))


def pretty_print_commercial_breaks(breaks: list[list[EdlEvent]]) -> str:
    def s_to_ts(t: float) -> str:
        hour = int(t / 3600.0)
        minute = int(t / 60) % 60
        second = t % 60.0
        return f"{hour:02d}:{minute:02d}:{second:06.3f}"

    result = ""
    for br in sorted(breaks, key=lambda b: b[0].start if b and b[0] else 0):
        for event in br:
            if event is None:
                result += " " * 27 + ", "
            else:
                result += f"{s_to_ts(event.start)} - {s_to_ts(event.end)}, "
        result += "\n"

    return result
