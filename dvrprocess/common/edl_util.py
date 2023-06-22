from enum import Enum


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

    def length(self):
        return max(0, self.end - self.start)


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


