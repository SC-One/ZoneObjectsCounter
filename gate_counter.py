"""
Gate IN/OUT counting logic.

Pure Python (stdlib only), no dependency on YOLO / ByteTrack / OpenCV / numpy.
Input per frame: a list of (track_id, (x1, y1, x2, y2)).
Output per frame: a list of GateEvent.

Algorithm (per track, independent state machine):
  IDLE     --reference point enters gate polygon--> IN_GATE  (remember entry point)
  IN_GATE  --net displacement along `in_direction` >= threshold--> COUNTED (emit IN/OUT)
  IN_GATE  --leaves polygon / track lost--> final decision, else back to IDLE
  COUNTED  --outside the gate for `rearm_frames_outside` frames--> IDLE (re-armed)
A track is counted at most once per gate passage, so it can't be double counted.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from typing import Deque, Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]

IDLE, IN_GATE, COUNTED = "IDLE", "IN_GATE", "COUNTED"


@dataclass
class GateConfig:
    polygon: List[Point]                      # gate / ROI polygon (>= 3 points), image pixels
    in_direction: Tuple[Point, Point]         # arrow p0 -> p1: walking this way = IN
    ref_point: str = "bottom_center"          # "bottom_center" (feet) | "center"
    smooth_window: int = 5                    # moving-average window on the reference point
    min_displacement_px: float = 30.0         # min travel along in_direction to decide...
    min_displacement_bbox_ratio: float = 0.5  # ...or this * median bbox height (max of both)
    min_alignment: float = 0.4                # |proj| / |displacement|; rejects lateral walking
    min_frames_in_gate: int = 3               # min frames inside before an early decision
    rearm_frames_outside: int = 10            # frames outside gate before a track can count again
    track_timeout_frames: int = 30            # drop track if unseen this many frames
    max_history: int = 300                    # trail length kept per track

    @classmethod
    def from_dict(cls, d: dict) -> "GateConfig":
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in names}
        kw["polygon"] = [(float(x), float(y)) for x, y in kw["polygon"]]
        a, b = kw["in_direction"]
        kw["in_direction"] = ((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
        return cls(**kw)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["polygon"] = [list(p) for p in self.polygon]
        d["in_direction"] = [list(p) for p in self.in_direction]
        return d


@dataclass
class GateEvent:
    type: str          # "IN" | "OUT"
    track_id: int
    frame: int
    point: Point       # smoothed reference point at decision time


@dataclass
class TrackState:
    track_id: int
    state: str = IDLE
    raw: Deque[Point] = field(default_factory=deque)
    trail: Deque[Point] = field(default_factory=deque)     # smoothed history (for display)
    heights: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    last_frame: int = -1
    inside: bool = False
    entry_pt: Optional[Point] = None
    frames_inside: int = 0
    frames_outside: int = 0
    last_event: Optional[str] = None


# ----------------------------------------------------------------- geometry
def point_in_polygon(p: Point, poly: Sequence[Point]) -> bool:
    x, y = p
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def reference_point(box: BBox, mode: str) -> Point:
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    return (cx, y2) if mode == "bottom_center" else (cx, 0.5 * (y1 + y2))


def _median(v: Sequence[float]) -> float:
    s = sorted(v)
    n = len(s)
    return 0.0 if n == 0 else (s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]))


# ------------------------------------------------------------------ counter
class GateCounter:
    def __init__(self, cfg: GateConfig):
        self.tracks: Dict[int, TrackState] = {}
        self.count_in = 0
        self.count_out = 0
        self.set_config(cfg)

    def set_config(self, cfg: GateConfig) -> None:
        self.cfg = cfg
        (x0, y0), (x1, y1) = cfg.in_direction
        dx, dy = x1 - x0, y1 - y0
        n = math.hypot(dx, dy) or 1.0
        self._dir = (dx / n, dy / n)

    def reset(self) -> None:
        self.tracks.clear()
        self.count_in = self.count_out = 0

    # ------------------------------------------------------------- public
    def update(self, frame_idx: int, detections: Sequence[Tuple[int, BBox]]) -> List[GateEvent]:
        events: List[GateEvent] = []
        seen = set()
        for tid, box in detections:
            if tid in seen:
                continue
            seen.add(tid)
            st = self.tracks.get(tid)
            if st is None:
                st = self.tracks[tid] = TrackState(
                    tid,
                    raw=deque(maxlen=self.cfg.smooth_window),
                    trail=deque(maxlen=self.cfg.max_history),
                )
            self._push(st, frame_idx, box)
            ev = self._step(st, frame_idx)
            if ev:
                events.append(ev)

        for tid in [t for t, s in self.tracks.items()
                    if frame_idx - s.last_frame > self.cfg.track_timeout_frames]:
            st = self.tracks.pop(tid)
            if st.state == IN_GATE and st.trail:       # lost while crossing -> final decision
                ev = self._decide(st, frame_idx)
                if ev:
                    events.append(ev)
        return events

    # ----------------------------------------------------------- internals
    def _push(self, st: TrackState, frame_idx: int, box: BBox) -> None:
        st.last_frame = frame_idx
        st.heights.append(max(1.0, box[3] - box[1]))
        st.raw.append(reference_point(box, self.cfg.ref_point))
        n = len(st.raw)
        st.trail.append((sum(p[0] for p in st.raw) / n, sum(p[1] for p in st.raw) / n))

    def _step(self, st: TrackState, frame_idx: int) -> Optional[GateEvent]:
        cfg = self.cfg
        cur = st.trail[-1]
        st.inside = point_in_polygon(cur, cfg.polygon)

        if st.state == IDLE:
            if st.inside:
                st.state = IN_GATE
                st.frames_inside = 1
                # baseline = last point seen outside (captures the approach), else current
                st.entry_pt = st.trail[-2] if len(st.trail) > 1 else cur
            return None

        if st.state == IN_GATE:
            if st.inside:
                st.frames_inside += 1
                if st.frames_inside >= cfg.min_frames_in_gate:
                    return self._decide(st, frame_idx)
                return None
            ev = self._decide(st, frame_idx)           # left the polygon: final decision
            if ev is None:
                st.state = IDLE
            return ev

        if st.state == COUNTED:                        # locked until it has left the gate
            if st.inside:
                st.frames_outside = 0
            else:
                st.frames_outside += 1
                if st.frames_outside >= cfg.rearm_frames_outside:
                    st.state = IDLE
                    st.last_event = None
        return None

    def _decide(self, st: TrackState, frame_idx: int) -> Optional[GateEvent]:
        cfg = self.cfg
        cur, ent = st.trail[-1], st.entry_pt
        dx, dy = cur[0] - ent[0], cur[1] - ent[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return None
        thr = max(cfg.min_displacement_px, cfg.min_displacement_bbox_ratio * _median(st.heights))
        proj = dx * self._dir[0] + dy * self._dir[1]
        if abs(proj) < thr or abs(proj) / dist < cfg.min_alignment:
            return None
        etype = "IN" if proj > 0 else "OUT"
        if etype == "IN":
            self.count_in += 1
        else:
            self.count_out += 1
        st.state = COUNTED
        st.frames_outside = 0
        st.last_event = etype
        return GateEvent(etype, st.track_id, frame_idx, cur)
