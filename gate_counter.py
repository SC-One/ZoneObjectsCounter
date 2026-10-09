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
     - walks into a portal and VANISHES there (side != portal.leads_to). No speed / time condition:
       visible = still on its side, gone from inside a portal = went to the hidden side,
     - becomes "far" inside a portal that has `far_percent` set: the person's bbox is smaller than
       that % of the portal rectangle (area by default, or height), i.e. far from the camera =
       beyond the gate, so it is already on the hidden side even though still visible,
     - moves from an inside zone to an outside zone (or back).
No global direction vector is used, so it is independent of camera angle / perspective.

Reference point: every line / zone may set its own `ref_point` (which point of the person's bbox is
tested against it, see REF_POINTS); if not set, the global `CounterConfig.ref_point` is used.
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
    ref_point: Optional[str] = None  # bbox point tested against this line (None = global ref_point)


@dataclass
class Zone:
    id: str
    kind: str                        # "portal" | "inside" | "outside"
    polygon: List[Point]
    leads_to: str = OUTSIDE          # portal only: hidden side behind it
    far_percent: float = 0.0         # portal only, 0 = off: person inside the portal whose bbox is smaller
                                     # than this % of the portal rectangle is on the hidden side
    far_metric: str = "area"         # "area" (bbox area / portal bounding-rect area) | "height"
    ref_point: Optional[str] = None  # bbox point tested against this zone (None = global ref_point)


@dataclass
class RegionConfig:
    zones: List[Zone] = field(default_factory=list)
    default_side: str = "auto"       # "auto" | "inside" | "outside": meaning of unmarked area
    min_dwell_frames: int = 3        # frames in a region before it counts as the track's region
    commit_lost_frames: int = 8      # lost this long while in a portal => vanished into it
    portal_use_box_center: bool = True   # portal also matches when the bbox CENTER is inside (partial/occluded bodies)
    min_travel_px: float = 30.0      # min movement to accept a visible->visible transition ...
    min_travel_bbox_ratio: float = 0.35  # ... = max(px, ratio * bbox height)


@dataclass
class CounterConfig:
    lines: List[LineGate] = field(default_factory=list)
    region: RegionConfig = field(default_factory=RegionConfig)
    ref_point: str = "bottom_center"     # default bbox point, see REF_POINTS (bottom_center = feet)
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
    trail: Deque[Point]                                   # smoothed history of the GLOBAL ref point
    raws: Dict[str, Deque[Point]] = field(default_factory=dict)   # per ref name: recent raw points
    pts: Dict[str, Point] = field(default_factory=dict)           # per ref name: smoothed current point
    heights: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    last_frame: int = -1
    sm_w: float = 0.0                                     # smoothed bbox size
    sm_h: float = 0.0
    size_ratio: Optional[float] = None                    # person size / portal rect size (debug)
    lines: Dict[str, LineState] = field(default_factory=dict)
    # region model
    reg_raw: Optional[Tuple[str, str, bool]] = None
    reg_run: int = 0
    reg_stable: Optional[str] = None
    stable_leads: str = OUTSIDE
    stable_far: bool = False
    last_reg: Optional[str] = None
    last_leads: str = OUTSIDE
    portal_id: str = "region"
    center: Optional[Point] = None                        # raw bbox center of the last box
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


REF_POINTS = {                    # name -> (fx, fy): fraction of bbox width / height from the top-left
    "bottom_center": (0.5, 1.0), "center": (0.5, 0.5), "top_center": (0.5, 0.0),
    "left_center": (0.0, 0.5), "right_center": (1.0, 0.5),
    "bottom_left": (0.0, 1.0), "bottom_right": (1.0, 1.0),
    "top_left": (0.0, 0.0), "top_right": (1.0, 0.0),
}
_REF_ALIASES = {"bottom": "bottom_center", "top": "top_center", "left": "left_center", "right": "right_center"}


def reference_point(box: BBox, mode: str) -> Point:
    name = _REF_ALIASES.get(mode, mode)
    if name not in REF_POINTS:
        raise ValueError(f"unknown ref_point {mode!r}, use one of {sorted(REF_POINTS)}")
    fx, fy = REF_POINTS[name]
    x1, y1, x2, y2 = box
    return (x1 + fx * (x2 - x1), y1 + fy * (y2 - y1))


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
                st = self.tracks[tid] = TrackState(tid, trail=deque(maxlen=cfg.max_history))
            self._push(st, frame_idx, box)
            st.vanish_checked = False
            cur, h = st.trail[-1], _median(st.heights)
            for lg in cfg.lines:
                t = self._line_step(st, lg, st.pts[lg.ref_point or cfg.ref_point], h)
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
            if rc.zones and not st.vanish_checked and missed >= min(rc.commit_lost_frames, cfg.track_timeout_frames):
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

    def classify(self, pts: Dict[str, Point], center: Optional[Point] = None,
                 size: Optional[Tuple[float, float]] = None) -> Tuple[str, str, str, bool, Optional[float]]:
        """-> (region 'portal'|'inside'|'outside', leads_to, zone_id, far, size_ratio)
        `pts` = smoothed person point per ref name (each zone uses its own ref_point or the global one).
        `center` = extra point tested for portals. `size` = (bbox w, h): if the portal has far_percent
        and the person is smaller than that, the region is the portal's hidden side (far=True)."""
        zs = self.cfg.region.zones
        for z in zs:
            if z.kind == PORTAL and (point_in_polygon(pts[z.ref_point or self.cfg.ref_point], z.polygon)
                                     or (center is not None and point_in_polygon(center, z.polygon))):
                xs = [p[0] for p in z.polygon]
                ys = [p[1] for p in z.polygon]
                rect_w, rect_h = max(xs) - min(xs), max(ys) - min(ys)
                ratio = None
                if size is not None and rect_w > 1e-6 and rect_h > 1e-6:
                    ratio = (size[1] / rect_h) if z.far_metric == "height" \
                        else (size[0] * size[1]) / (rect_w * rect_h)
                if z.far_percent > 0 and ratio is not None and ratio < z.far_percent / 100.0:
                    return z.leads_to, z.leads_to, z.id, True, ratio
                return PORTAL, z.leads_to, z.id, False, ratio
        for z in zs:
            if z.kind in (INSIDE, OUTSIDE) and point_in_polygon(pts[z.ref_point or self.cfg.ref_point], z.polygon):
                return z.kind, z.kind, z.id, False, None
        d = self.default_side()
        return d, d, "region", False, None

    # ------------------------------------------------------------ internals
    def _push(self, st: TrackState, frame_idx: int, box: BBox) -> None:
        st.last_frame = frame_idx
        w, h = max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])
        st.sm_w, st.sm_h = (w, h) if st.sm_h == 0.0 else (0.6 * st.sm_w + 0.4 * w, 0.6 * st.sm_h + 0.4 * h)
        st.heights.append(h)
        st.center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
        for name in self._ref_names():                # one smoothed point per ref name in use
            dq = st.raws.get(name)
            if dq is None:
                dq = st.raws[name] = deque(maxlen=self.cfg.smooth_window)
            dq.append(reference_point(box, name))
            n = len(dq)
            st.pts[name] = (sum(p[0] for p in dq) / n, sum(p[1] for p in dq) / n)
        st.trail.append(st.pts[self.cfg.ref_point])

    def _ref_names(self) -> set:
        cfg = self.cfg
        names = {cfg.ref_point}
        names.update(l.ref_point for l in cfg.lines if l.ref_point)
        names.update(z.ref_point for z in cfg.region.zones if z.ref_point)
        return names

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
        reg, leads, zid, far, ratio = self.classify(
            st.pts, st.center if rc.portal_use_box_center else None, (st.sm_w, st.sm_h))
        st.size_ratio = ratio
        if reg == PORTAL or far:
            st.portal_id = zid
        if reg == PORTAL and st.reg_raw is None:      # first seen inside a portal => came from hidden side
            st.side, st.side_pt = leads, cur
        st.last_reg, st.last_leads = reg, leads
        key = (reg, leads, far)
        if key == st.reg_raw:
            st.reg_run += 1
        else:
            st.reg_raw, st.reg_run = key, 1
        if st.reg_run >= rc.min_dwell_frames:
            st.reg_stable, st.stable_leads, st.stable_far = reg, leads, far
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
        # "far" inside a portal is decided by size (walking away from the camera moves few pixels);
        # visible -> visible transitions need real travel
        if st.stable_far or _dist(cur, st.side_pt) >= self._thr(h, rc.min_travel_px, rc.min_travel_bbox_ratio):
            st.side, st.side_pt = S, cur
            return "IN" if S == INSIDE else "OUT"
        return None

    def _vanish(self, st: TrackState) -> Optional[str]:
        """Called once the track has been absent `commit_lost_frames` frames (only a debounce against
        1-2 frame detector dropouts). No condition on speed or on how long the person was there:
        last seen inside a portal and known side != hidden side  =>  went to the hidden side."""
        if st.last_reg != PORTAL or st.side is None or st.side == st.last_leads:
            return None
        st.side, st.side_pt = st.last_leads, st.trail[-1]
        return "OUT" if st.last_leads == OUTSIDE else "IN"
