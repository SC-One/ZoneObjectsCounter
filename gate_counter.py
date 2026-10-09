"""
People IN/OUT counting logic.  Pure Python (stdlib only) - no YOLO / ByteTrack / OpenCV / numpy.

Input  per frame : [(track_id, (x1, y1, x2, y2)), ...]      (pixel coordinates)
Output per frame : [GateEvent, ...]

Two independent gate types (use either, or both, any number of lines):

1) LineGate  - a virtual line on the floor. Each track keeps the side of the line it is on
   (with a dead-band, so jitter can't flip it). A side flip, within the line's extent,
   is a crossing. `in_side` ("left"/"right" of p0->p1 as seen on screen) says which side is INSIDE.
   Use when you can see both sides of the passage.

2) Region model (zones) - people are classified by WHERE they are, not by a direction:
     portal  : a "commit zone" where people appear / vanish (a door seen head-on, a corridor
               mouth...). `leads_to` = what is hidden behind it ("outside" or "inside").
     inside / outside : optional visible zones. Everything not covered = `default_side`
               (auto: the opposite of what the portal leads to).
   Each track has a `side`. It changes (=> event) when the track:
     - leaves a portal into a visible area on the other side than the portal leads to,
     - walks into a portal and VANISHES there (side != portal.leads_to),
     - moves from an inside zone to an outside zone (or back).
No global direction vector is used, so it is independent of camera angle / perspective.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from typing import Deque, Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
INSIDE, OUTSIDE, PORTAL = "inside", "outside", "portal"


# ------------------------------------------------------------------ config
@dataclass
class LineGate:
    id: str
    p0: Point
    p1: Point
    in_side: str = "left"            # side (looking p0->p1 on screen) that is INSIDE: "left"|"right"
    margin_px: float = 8.0           # dead-band half-width ...
    margin_bbox_ratio: float = 0.08  # ... = max(px, ratio * bbox height)  (perspective-aware)
    extend: float = 0.15             # crossing accepted this fraction beyond the segment ends


@dataclass
class Zone:
    id: str
    kind: str                        # "portal" | "inside" | "outside"
    polygon: List[Point]
    leads_to: str = OUTSIDE          # portal only: hidden side behind it


@dataclass
class RegionConfig:
    zones: List[Zone] = field(default_factory=list)
    default_side: str = "auto"       # "auto" | "inside" | "outside": meaning of unmarked area
    min_dwell_frames: int = 3        # frames in a region before it counts as the track's region
    commit_lost_frames: int = 8      # lost this long while in a portal => vanished into it
    vanish_lookback_frames: int = 30 # window used to check the track was really moving
    min_travel_px: float = 30.0      # min movement to accept a transition ...
    min_travel_bbox_ratio: float = 0.35  # ... = max(px, ratio * bbox height)


@dataclass
class CounterConfig:
    lines: List[LineGate] = field(default_factory=list)
    region: RegionConfig = field(default_factory=RegionConfig)
    ref_point: str = "bottom_center"     # "bottom_center" (feet) | "center"
    smooth_window: int = 5
    track_timeout_frames: int = 30
    max_history: int = 300

    @classmethod
    def from_dict(cls, d: dict) -> "CounterConfig":
        def build(c, x):
            names = {f.name for f in fields(c)}
            return c(**{k: v for k, v in x.items() if k in names})
        d = dict(d)
        lines = [build(LineGate, {**l, "p0": tuple(l["p0"]), "p1": tuple(l["p1"])})
                 for l in d.pop("lines", [])]
        r = dict(d.pop("region", {}))
        zones = [build(Zone, {**z, "polygon": [tuple(p) for p in z["polygon"]]})
                 for z in r.pop("zones", [])]
        return build(cls, {**d, "lines": lines, "region": build(RegionConfig, {**r, "zones": zones})})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GateEvent:
    type: str          # "IN" | "OUT"
    track_id: int
    frame: int
    point: Point
    gate_id: str


# ------------------------------------------------------------------ state
@dataclass
class LineState:
    side: int = 0
    pt: Optional[Point] = None


@dataclass
class TrackState:
    track_id: int
    raw: Deque[Point]
    trail: Deque[Point]                                   # smoothed history (for display)
    heights: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    last_frame: int = -1
    lines: Dict[str, LineState] = field(default_factory=dict)
    # region model
    reg_raw: Optional[Tuple[str, str]] = None
    reg_run: int = 0
    reg_stable: Optional[str] = None
    stable_leads: str = OUTSIDE
    last_reg: Optional[str] = None
    last_leads: str = OUTSIDE
    portal_id: str = "region"
    side: Optional[str] = None                            # inside/outside, None = unknown
    side_pt: Optional[Point] = None
    vanish_checked: bool = False
    last_event: Optional[str] = None


# ----------------------------------------------------------------- helpers
def point_in_polygon(p: Point, poly: Sequence[Point]) -> bool:
    x, y = p
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
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


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _opp(s: str) -> str:
    return OUTSIDE if s == INSIDE else INSIDE


# ----------------------------------------------------------------- counter
class GateCounter:
    def __init__(self, cfg: CounterConfig):
        self.cfg = cfg
        self.tracks: Dict[int, TrackState] = {}
        self.counts: Dict[str, Dict[str, int]] = {}      # per gate id
        self.count_in = 0
        self.count_out = 0

    def set_config(self, cfg: CounterConfig) -> None:
        self.cfg = cfg

    def reset(self) -> None:
        self.tracks.clear()
        self.counts.clear()
        self.count_in = self.count_out = 0

    # --------------------------------------------------------------- public
    def update(self, frame_idx: int, detections: Sequence[Tuple[int, BBox]]) -> List[GateEvent]:
        cfg, rc = self.cfg, self.cfg.region
        events: List[GateEvent] = []
        seen = set()
        for tid, box in detections:
            if tid in seen:
                continue
            seen.add(tid)
            st = self.tracks.get(tid)
            if st is None:
                st = self.tracks[tid] = TrackState(
                    tid, raw=deque(maxlen=cfg.smooth_window), trail=deque(maxlen=cfg.max_history))
            self._push(st, frame_idx, box)
            st.vanish_checked = False
            cur, h = st.trail[-1], _median(st.heights)
            for lg in cfg.lines:
                t = self._line_step(st, lg, cur, h)
                if t:
                    events.append(self._emit(st, t, lg.id, frame_idx, cur))
            if rc.zones:
                t = self._region_step(st, cur, h)
                if t:
                    events.append(self._emit(st, t, st.portal_id, frame_idx, cur))

        for tid, st in list(self.tracks.items()):
            if tid in seen:
                continue
            missed = frame_idx - st.last_frame
            if rc.zones and not st.vanish_checked and missed >= rc.commit_lost_frames:
                st.vanish_checked = True
                t = self._vanish(st)
                if t:
                    events.append(self._emit(st, t, st.portal_id, frame_idx, st.trail[-1]))
            if missed > cfg.track_timeout_frames:
                del self.tracks[tid]
        return events

    def default_side(self) -> str:
        """Meaning of the unmarked (visible) area."""
        rc = self.cfg.region
        if rc.default_side in (INSIDE, OUTSIDE):
            return rc.default_side
        kinds = {z.kind for z in rc.zones}
        if INSIDE in kinds and OUTSIDE not in kinds:
            return OUTSIDE
        if OUTSIDE in kinds and INSIDE not in kinds:
            return INSIDE
        for z in rc.zones:
            if z.kind == PORTAL:
                return _opp(z.leads_to)
        return INSIDE

    def classify(self, pt: Point) -> Tuple[str, str, str]:
        """-> (region 'portal'|'inside'|'outside', leads_to, zone_id)"""
        zs = self.cfg.region.zones
        for z in zs:
            if z.kind == PORTAL and point_in_polygon(pt, z.polygon):
                return PORTAL, z.leads_to, z.id
        for z in zs:
            if z.kind in (INSIDE, OUTSIDE) and point_in_polygon(pt, z.polygon):
                return z.kind, z.kind, z.id
        d = self.default_side()
        return d, d, "region"

    # ------------------------------------------------------------ internals
    def _push(self, st: TrackState, frame_idx: int, box: BBox) -> None:
        st.last_frame = frame_idx
        st.heights.append(max(1.0, box[3] - box[1]))
        st.raw.append(reference_point(box, self.cfg.ref_point))
        n = len(st.raw)
        st.trail.append((sum(p[0] for p in st.raw) / n, sum(p[1] for p in st.raw) / n))

    def _emit(self, st: TrackState, etype: str, gate_id: str, frame_idx: int, pt: Point) -> GateEvent:
        if etype == "IN":
            self.count_in += 1
        else:
            self.count_out += 1
        self.counts.setdefault(gate_id, {"IN": 0, "OUT": 0})[etype] += 1
        st.last_event = etype
        return GateEvent(etype, st.track_id, frame_idx, pt, gate_id)

    @staticmethod
    def _thr(h: float, px: float, ratio: float) -> float:
        return max(px, ratio * h)

    # ---- line gate
    def _line_step(self, st: TrackState, lg: LineGate, cur: Point, h: float) -> Optional[str]:
        dx, dy = lg.p1[0] - lg.p0[0], lg.p1[1] - lg.p0[1]
        L = math.hypot(dx, dy)
        if L < 1e-6:
            return None
        nx, ny = dy / L, -dx / L                      # unit normal pointing to the screen-LEFT
        d = (cur[0] - lg.p0[0]) * nx + (cur[1] - lg.p0[1]) * ny
        if abs(d) < max(lg.margin_px, lg.margin_bbox_ratio * h):
            return None                               # dead-band: ignore jitter near the line
        side = 1 if d > 0 else -1
        ls = st.lines.setdefault(lg.id, LineState())
        if ls.side == 0 or side == ls.side:
            ls.side, ls.pt = side, cur
            return None
        d0 = (ls.pt[0] - lg.p0[0]) * nx + (ls.pt[1] - lg.p0[1]) * ny
        t = d0 / (d0 - d)                             # where the move ls.pt->cur hits the line
        ix = ls.pt[0] + t * (cur[0] - ls.pt[0])
        iy = ls.pt[1] + t * (cur[1] - ls.pt[1])
        u = ((ix - lg.p0[0]) * dx + (iy - lg.p0[1]) * dy) / (L * L)
        ls.side, ls.pt = side, cur
        if u < -lg.extend or u > 1.0 + lg.extend:
            return None                               # went around the end of the line
        return "IN" if side == (1 if lg.in_side == "left" else -1) else "OUT"

    # ---- region model
    def _region_step(self, st: TrackState, cur: Point, h: float) -> Optional[str]:
        rc = self.cfg.region
        reg, leads, zid = self.classify(cur)
        if reg == PORTAL:
            st.portal_id = zid
            if st.reg_raw is None:                    # first seen inside a portal => came from hidden side
                st.side, st.side_pt = leads, cur
        st.last_reg, st.last_leads = reg, leads
        key = (reg, leads)
        if key == st.reg_raw:
            st.reg_run += 1
        else:
            st.reg_raw, st.reg_run = key, 1
        if st.reg_run >= rc.min_dwell_frames:
            st.reg_stable, st.stable_leads = reg, leads
        S = st.reg_stable
        if S is None:
            return None
        if S == PORTAL:
            if st.side is None or st.side == st.stable_leads:
                st.side, st.side_pt = st.stable_leads, cur
            return None                               # came from a visible side: keep side_pt
        if st.side is None or S == st.side:
            st.side, st.side_pt = S, cur
            return None
        if _dist(cur, st.side_pt) >= self._thr(h, rc.min_travel_px, rc.min_travel_bbox_ratio):
            st.side, st.side_pt = S, cur
            return "IN" if S == INSIDE else "OUT"
        return None

    def _vanish(self, st: TrackState) -> Optional[str]:
        rc = self.cfg.region
        if st.last_reg != PORTAL or st.side is None or st.side == st.last_leads:
            return None
        pts = list(st.trail)[-rc.vanish_lookback_frames:]
        if _dist(pts[0], pts[-1]) < self._thr(_median(st.heights), rc.min_travel_px, rc.min_travel_bbox_ratio):
            return None                               # was not really walking into the portal
        st.side, st.side_pt = st.last_leads, st.trail[-1]
        return "OUT" if st.last_leads == OUTSIDE else "IN"
