"""
People IN/OUT counter - debug GUI.   YOLO + ByteTrack (ultralytics) -> GateCounter (independent logic)

Gates (draw any time, video freezes on the current frame while drawing):
  l  LINE   : click start, end, then a point on the INSIDE side of the line
  p  PORTAL : polygon (Enter) where people appear/vanish (door seen head-on...), then O/I = what is
              hidden behind it (O = outside, I = inside)
  o  OUTSIDE zone polygon      i  INSIDE zone polygon      (Enter to finish, Backspace undo)
Other keys:
  x undo last gate | c clear all gates | f unmarked-area meaning (auto/inside/outside)
  [ ] lower / raise far_percent (0 = off) of the last portal, m = area/height measure; s saves it
  y cycle the ref point of the last gate (global/bottom/center/top/left/right) | w toggle fullscreen | s save config | r reset counts | space pause | t trails | d debug | h help | q/Esc quit
"""
import argparse
import json
import math
import time
from collections import deque
from dataclasses import asdict

import cv2
import numpy as np

from gate_counter import CounterConfig, GateCounter, LineGate, Zone

KIND_COLOR = {"inside": (0, 200, 0), "outside": (255, 120, 0), "portal": (0, 165, 255)}
LINE_COLOR = (255, 0, 255)


# ------------------------------------------------------------ normalized <-> pixel config
def scale_counter_dict(d, sx, sy, nd=None):
    d = json.loads(json.dumps(d))
    f = (lambda v: round(v, nd)) if nd is not None else (lambda v: v)
    for l in d.get("lines", []):
        l["p0"] = [f(l["p0"][0] * sx), f(l["p0"][1] * sy)]
        l["p1"] = [f(l["p1"][0] * sx), f(l["p1"][1] * sy)]
    for z in d.get("region", {}).get("zones", []):
        z["polygon"] = [[f(x * sx), f(y * sy)] for x, y in z["polygon"]]
    return d


# ------------------------------------------------------------ detection
class PersonTracker:
    """frame -> [(track_id, (x1,y1,x2,y2))]. The only place that knows YOLO / ByteTrack."""

    def __init__(self, cfg):
        from ultralytics import YOLO
        self.cfg = cfg
        self.model = YOLO(cfg["model"])

    def __call__(self, frame):
        c = self.cfg
        r = self.model.track(frame, persist=True, tracker=c["tracker"], classes=[0],
                             conf=c["conf"], iou=c["iou"], imgsz=c["imgsz"],
                             device=c.get("device"), verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            return []
        ids = r.boxes.id.int().cpu().tolist()
        xyxy = r.boxes.xyxy.cpu().tolist()
        return [(i, tuple(b)) for i, b in zip(ids, xyxy)]


# ------------------------------------------------------------ gate editor
class Editor:
    def __init__(self, scale):
        self.scale = scale
        self.mode = None          # None | "line" | "poly"
        self.kind = None
        self.stage = ""
        self.pts = []
        self.result = None

    @property
    def active(self):
        return self.mode is not None

    def start_line(self):
        self.mode, self.kind, self.stage, self.pts, self.result = "line", "line", "points", [], None

    def start_poly(self, kind):
        self.mode, self.kind, self.stage, self.pts, self.result = "poly", kind, "points", [], None

    def on_mouse(self, event, x, y, *_):
        if not self.active or event != cv2.EVENT_LBUTTONDOWN or self.stage != "points":
            return
        self.pts.append((x / self.scale, y / self.scale))
        if self.mode == "line" and len(self.pts) == 3:
            p0, p1, q = self.pts
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            s = (q[0] - p0[0]) * dy - (q[1] - p0[1]) * dx          # >0 => q is on the screen-left
            self.result = {"type": "line", "p0": p0, "p1": p1, "in_side": "left" if s > 0 else "right"}
            self.mode = None

    def _finish_poly(self, leads="outside"):
        self.result = {"type": "zone", "kind": self.kind, "polygon": self.pts, "leads_to": leads}
        self.mode = None

    def key(self, k):
        if k == 27:
            self.mode = None
        elif k in (8, 127) and self.stage == "points" and self.pts:
            self.pts.pop()
        elif self.mode == "poly" and self.stage == "points" and k in (13, 10) and len(self.pts) >= 3:
            if self.kind == "portal":
                self.stage = "leads"
            else:
                self._finish_poly()
        elif self.mode == "poly" and self.stage == "leads":
            if k == ord("o"):
                self._finish_poly("outside")
            elif k == ord("i"):
                self._finish_poly("inside")

    def draw(self, img):
        s = self.scale
        pts = [(int(x * s), int(y * s)) for x, y in self.pts]
        col = LINE_COLOR if self.mode == "line" else KIND_COLOR[self.kind]
        for p in pts:
            cv2.circle(img, p, 5, col, -1)
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts[:2] if self.mode == "line" else pts)], False, col, 2)
        if self.mode == "line":
            msg = ["LINE: click START point", "LINE: click END point",
                   "LINE: click a point on the INSIDE side"][len(self.pts)]
        elif self.stage == "leads":
            msg = "PORTAL: what is hidden behind it?  O = outside   I = inside"
        else:
            msg = f"{self.kind.upper()} zone: click points, Enter = finish, Backspace = undo"
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(img, "DRAWING - " + msg + "   (Esc = cancel)", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def uid(prefix, existing):
    n = 1
    while f"{prefix}{n}" in existing:
        n += 1
    return f"{prefix}{n}"


def apply_result(counter, res, added):
    cfg = counter.cfg
    if res["type"] == "line":
        obj = LineGate(id=uid("line", {l.id for l in cfg.lines}), p0=res["p0"], p1=res["p1"],
                       in_side=res["in_side"])
        cfg.lines.append(obj)
        added.append(("line", obj))
    else:
        obj = Zone(id=uid(res["kind"], {z.id for z in cfg.region.zones}), kind=res["kind"],
                   polygon=res["polygon"], leads_to=res["leads_to"])
        cfg.region.zones.append(obj)
        added.append(("zone", obj))
    counter.reset()


def undo_last(counter, added):
    if not added:
        return
    kind, obj = added.pop()
    (counter.cfg.lines if kind == "line" else counter.cfg.region.zones).remove(obj)
    counter.reset()


# ------------------------------------------------------------ drawing
def color_for(i):
    rng = np.random.RandomState(i * 7919 % 2**31)
    return tuple(int(v) for v in rng.randint(60, 255, 3))


def draw_all(frame, counter, dets, flashes, log, scale, show_trails, show_debug, frame_idx, show_help):
    img = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame.copy()
    S = lambda p: (int(p[0] * scale), int(p[1] * scale))
    cfg = counter.cfg

    # zones
    for z in cfg.region.zones:
        poly = np.array([S(p) for p in z.polygon])
        col = KIND_COLOR[z.kind]
        ov = img.copy()
        cv2.fillPoly(ov, [poly], col)
        img = cv2.addWeighted(ov, 0.2, img, 0.8, 0)
        cv2.polylines(img, [poly], True, col, 2)
        label = f"PORTAL ({z.id}) -> hidden: {z.leads_to}" if z.kind == "portal" else f"{z.kind.upper()} ({z.id})"
        if z.ref_point:
            label += f" ref={z.ref_point}"
        c = poly.mean(axis=0).astype(int)
        cv2.putText(img, label, (int(c[0]) - 60, int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        if z.kind == "portal" and z.far_percent > 0:
            # reference box: a person smaller than this box (inside the portal) is "far" = hidden side
            xs = [p[0] for p in z.polygon]
            ys = [p[1] for p in z.polygon]
            rw, rh = max(xs) - min(xs), max(ys) - min(ys)
            if z.far_metric == "height":
                th = rh * z.far_percent / 100.0
            else:
                th = math.sqrt(rw * rh * z.far_percent / 100.0 / 0.4)    # person box ~ 0.4 x 1 aspect
            bx, by = max(xs) + 12 / scale, max(ys)
            cv2.rectangle(img, S((bx, by - th)), S((bx + 0.4 * th, by)), col, 2)
            cv2.putText(img, f"far < {z.far_percent:g}% ({z.far_metric})", S((bx, by - th - 6 / scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    # lines + IN arrow (perpendicular, pointing to the INSIDE side)
    for l in cfg.lines:
        cv2.line(img, S(l.p0), S(l.p1), LINE_COLOR, 3)
        dx, dy = l.p1[0] - l.p0[0], l.p1[1] - l.p0[1]
        L = math.hypot(dx, dy) or 1.0
        nx, ny = dy / L, -dx / L
        if l.in_side == "right":
            nx, ny = -nx, -ny
        m = ((l.p0[0] + l.p1[0]) / 2, (l.p0[1] + l.p1[1]) / 2)
        tip = (m[0] + nx * 60 / scale, m[1] + ny * 60 / scale)
        cv2.arrowedLine(img, S(m), S(tip), LINE_COLOR, 2, tipLength=0.3)
        cv2.putText(img, f"{l.id}: IN side" + (f" ref={l.ref_point}" if l.ref_point else ""), S(tip),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, LINE_COLOR, 2)

    if show_trails:
        for tid, st in counter.tracks.items():
            if len(st.trail) > 1:
                cv2.polylines(img, [np.array([S(p) for p in st.trail])], False, color_for(tid), 2)

    for tid, (x1, y1, x2, y2) in dets:
        c = color_for(tid)
        cv2.rectangle(img, S((x1, y1)), S((x2, y2)), c, 2)
        st = counter.tracks.get(tid)
        label = f"ID {tid}"
        if show_debug and st and st.side:
            label += f" [{st.side}]"
        if show_debug and st and st.size_ratio is not None:
            label += f" {int(st.size_ratio * 100)}%"      # person size / portal rect size (far_metric)
        cv2.putText(img, label, (int(x1 * scale), int(y1 * scale) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
        if st:                                        # rings = the ref point of gates that use their own
            for g, gcol in [(l, LINE_COLOR) for l in cfg.lines] + [(z, KIND_COLOR[z.kind]) for z in cfg.region.zones]:
                if g.ref_point and g.ref_point != cfg.ref_point and g.ref_point in st.pts:
                    cv2.circle(img, S(st.pts[g.ref_point]), 7, gcol, 2)
        if st and st.trail:
            dot = KIND_COLOR.get(st.last_reg, (0, 255, 255)) if st.last_reg else (0, 255, 255)
            cv2.circle(img, S(st.trail[-1]), 5, dot, -1)

    for f_idx, ev in list(flashes):
        if frame_idx - f_idx > 45:
            flashes.remove((f_idx, ev))
            continue
        col = (0, 220, 0) if ev.type == "IN" else (0, 0, 255)
        cv2.putText(img, f"{ev.type} #{ev.track_id}", S((ev.point[0], ev.point[1] - 40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)

    # HUD
    n = min(len(log), 6)
    cv2.rectangle(img, (0, 0), (340, 62 + 22 * n), (0, 0, 0), -1)
    cv2.putText(img, f"IN: {counter.count_in}   OUT: {counter.count_out}", (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if cfg.region.zones:
        cv2.putText(img, f"unmarked area = {counter.default_side()}", (8, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    for i, (f_idx, ev) in enumerate(list(log)[:6]):
        cv2.putText(img, f"f{f_idx}: {ev.type} id={ev.track_id} [{ev.gate_id}]", (8, 76 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0) if ev.type == "IN" else (80, 80, 255), 1)

    if not cfg.lines and not cfg.region.zones:
        cv2.putText(img, "No gates yet: press L (line), P (portal), O / I (outside / inside zone)",
                    (20, img.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if show_help:
        cv2.putText(img, "l line | p portal | o outside zone | i inside zone | x undo | c clear | "
                         "f default side | y ref point (last gate) | [ ] far% m metric | w fullscreen | s save | r reset | h help",
                    (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--source", help="camera index (0), video path or RTSP url; overrides config")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    src = str(args.source if args.source is not None else cfg["source"])
    src = int(src) if src.isdigit() else src
    is_file = not isinstance(src, int) and not str(src).startswith(("rtsp", "http"))

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {src}")
    if not is_file:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not w or not h:
        ok, probe = cap.read()
        if not ok:
            raise SystemExit("Cannot read from source")
        h, w = probe.shape[:2]
        if is_file:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    tracker = PersonTracker(cfg)
    counter = GateCounter(CounterConfig.from_dict(scale_counter_dict(cfg.get("counter", {}), w, h)))
    added = [("line", l) for l in counter.cfg.lines] + [("zone", z) for z in counter.cfg.region.zones]
    scale = float(cfg.get("display_scale", 1.0))
    editor = Editor(scale)

    win = "People IN/OUT"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)            # resizable window (drag its corner)
    fullscreen = bool(cfg.get("fullscreen", False))
    if fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(win, editor.on_mouse)

    frame, dets, frame_idx = None, [], 0
    flashes, log = [], deque(maxlen=20)
    paused, show_trails, show_debug, show_help = False, True, True, True

    while True:
        t0 = time.time()
        if not paused and not editor.active:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            dets = tracker(frame)
            for ev in counter.update(frame_idx, dets):
                print(f"[frame {ev.frame}] {ev.type} track_id={ev.track_id} gate={ev.gate_id}")
                flashes.append((frame_idx, ev))
                log.appendleft((frame_idx, ev))

        if editor.result:
            apply_result(counter, editor.result, added)
            editor.result = None
            flashes.clear(); log.clear()

        img = draw_all(frame, counter, dets, flashes, log, scale, show_trails, show_debug, frame_idx, show_help)
        if editor.active:
            editor.draw(img)
        cv2.imshow(win, img)

        delay = 1
        if is_file and cfg.get("realtime_playback", True) and not paused:
            delay = max(1, int(1000 / fps - (time.time() - t0) * 1000))
        k = cv2.waitKey(delay) & 0xFF
        if editor.active:
            editor.key(k)
            continue
        if k in (ord("q"), 27):
            break
        elif k == ord(" "):
            paused = not paused
        elif k == ord("l") and frame is not None:
            editor.start_line()
        elif k == ord("p") and frame is not None:
            editor.start_poly("portal")
        elif k == ord("o") and frame is not None:
            editor.start_poly("outside")
        elif k == ord("i") and frame is not None:
            editor.start_poly("inside")
        elif k == ord("x"):
            undo_last(counter, added); flashes.clear(); log.clear()
        elif k == ord("c"):
            counter.cfg.lines.clear(); counter.cfg.region.zones.clear(); added.clear()
            counter.reset(); flashes.clear(); log.clear()
        elif k in (ord("["), ord("]")):
            portals = [z for z in counter.cfg.region.zones if z.kind == "portal"]
            if portals:
                z = portals[-1]
                z.far_percent = float(min(100, max(0, z.far_percent + (5 if k == ord("]") else -5))))
                print(f"portal {z.id}: far_percent = {z.far_percent:g}" + ("  (off)" if z.far_percent == 0 else ""))
        elif k == ord("y") and added:
            g = added[-1][1]                          # last gate: cycle its own ref point
            order = [None, "bottom_center", "center", "top_center", "left_center", "right_center"]
            i = order.index(g.ref_point) if g.ref_point in order else -1
            g.ref_point = order[(i + 1) % len(order)]
            print(f"{g.id}: ref_point = {g.ref_point or 'global (' + counter.cfg.ref_point + ')'}")
            counter.reset(); flashes.clear(); log.clear()
        elif k == ord("w"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        elif k == ord("m"):
            portals = [z for z in counter.cfg.region.zones if z.kind == "portal"]
            if portals:
                z = portals[-1]
                z.far_metric = "height" if z.far_metric == "area" else "area"
                print(f"portal {z.id}: far_metric = {z.far_metric}")
        elif k == ord("f"):
            rc = counter.cfg.region
            rc.default_side = {"auto": "inside", "inside": "outside", "outside": "auto"}[rc.default_side]
            print("unmarked area =", counter.default_side(), f"(setting: {rc.default_side})")
        elif k == ord("s"):
            cfg["counter"] = scale_counter_dict(asdict(counter.cfg), 1.0 / w, 1.0 / h, nd=5)
            with open(args.config, "w") as f:
                json.dump(cfg, f, indent=2)
            print("Saved (normalized) to", args.config)
        elif k == ord("r"):
            counter.reset(); flashes.clear(); log.clear()
        elif k == ord("t"):
            show_trails = not show_trails
        elif k == ord("d"):
            show_debug = not show_debug
        elif k == ord("h"):
            show_help = not show_help

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final: IN={counter.count_in} OUT={counter.count_out}  per gate: {counter.counts}")


if __name__ == "__main__":
    main()
