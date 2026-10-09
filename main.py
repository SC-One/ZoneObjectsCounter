"""
Gate IN/OUT debug GUI.  YOLO + ByteTrack (ultralytics) -> GateCounter (independent logic).

Keys: q/Esc quit | space pause | g edit gate | s save config | r reset counts
      t toggle trails | d toggle debug labels
Gate editor: left-click polygon points, Backspace undo, Enter to finish polygon,
             then click 2 points = IN direction arrow (start -> end). Esc cancels.
"""
import argparse
import json
import time
from collections import deque

import cv2
import numpy as np

from gate_counter import GateConfig, GateCounter


# ------------------------------------------------------------ detection
class PersonTracker:
    """Thin wrapper: frame -> [(track_id, (x1,y1,x2,y2))]. Only place that knows YOLO/ByteTrack."""

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
class GateEditor:
    def __init__(self, scale):
        self.scale = scale
        self.active = False
        self.poly, self.arrow = [], []
        self.result = None

    def start(self):
        self.active, self.poly, self.arrow, self.result = True, [], [], None

    def on_mouse(self, event, x, y, *_):
        if not self.active or event != cv2.EVENT_LBUTTONDOWN:
            return
        p = (x / self.scale, y / self.scale)
        if self.result is None and not self._poly_done:
            self.poly.append(p)
        else:
            self.arrow.append(p)
            if len(self.arrow) == 2:
                self.result = (self.poly, tuple(self.arrow))
                self.active = False

    _poly_done = False

    def key(self, k):
        if k == 27:
            self.active = False
        elif k in (8, 127) and self.poly and not self._poly_done:
            self.poly.pop()
        elif k in (13, 10) and len(self.poly) >= 3:
            self._poly_done = True

    def start_fresh(self):
        self.start()
        self._poly_done = False

    def draw(self, img):
        s = self.scale
        pts = [(int(x * s), int(y * s)) for x, y in self.poly]
        for p in pts:
            cv2.circle(img, p, 4, (0, 255, 255), -1)
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts)], self._poly_done, (0, 255, 255), 2)
        for p in self.arrow:
            cv2.circle(img, (int(p[0] * s), int(p[1] * s)), 5, (255, 0, 255), -1)
        msg = ("Click IN-direction START then END" if self._poly_done
               else "Click polygon points, Enter=done, Backspace=undo, Esc=cancel")
        cv2.putText(img, "EDIT GATE: " + msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


# ------------------------------------------------------------ drawing
def color_for(i):
    rng = np.random.RandomState(i * 7919 % 2**31)
    return tuple(int(v) for v in rng.randint(60, 255, 3))


def draw_all(frame, counter, dets, flashes, log, scale, show_trails, show_debug, frame_idx):
    img = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame.copy()
    S = lambda p: (int(p[0] * scale), int(p[1] * scale))
    cfg = counter.cfg

    # gate overlay + direction arrow
    poly = np.array([S(p) for p in cfg.polygon])
    ov = img.copy()
    cv2.fillPoly(ov, [poly], (0, 200, 255))
    img = cv2.addWeighted(ov, 0.2, img, 0.8, 0)
    cv2.polylines(img, [poly], True, (0, 200, 255), 2)
    cv2.arrowedLine(img, S(cfg.in_direction[0]), S(cfg.in_direction[1]), (255, 0, 255), 3, tipLength=0.15)
    cv2.putText(img, "IN", S(cfg.in_direction[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # trails
    if show_trails:
        for tid, st in counter.tracks.items():
            if len(st.trail) > 1:
                cv2.polylines(img, [np.array([S(p) for p in st.trail])], False, color_for(tid), 2)

    # boxes / ids
    for tid, (x1, y1, x2, y2) in dets:
        c = color_for(tid)
        cv2.rectangle(img, S((x1, y1)), S((x2, y2)), c, 2)
        st = counter.tracks.get(tid)
        label = f"ID {tid}"
        if show_debug and st:
            label += f" {st.state}" + (f" {st.last_event}" if st.last_event else "")
        cv2.putText(img, label, (int(x1 * scale), int(y1 * scale) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
        if st and st.trail:
            cv2.circle(img, S(st.trail[-1]), 4, (0, 255, 0) if st.inside else (0, 0, 255), -1)

    # event flashes near the event position
    for f_idx, ev in list(flashes):
        if frame_idx - f_idx > 45:
            flashes.remove((f_idx, ev))
            continue
        col = (0, 220, 0) if ev.type == "IN" else (0, 0, 255)
        cv2.putText(img, f"{ev.type} #{ev.track_id}", S((ev.point[0], ev.point[1] - 40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)

    # HUD
    cv2.rectangle(img, (0, 0), (260, 40 + 22 * min(len(log), 6)), (0, 0, 0), -1)
    cv2.putText(img, f"IN: {counter.count_in}   OUT: {counter.count_out}", (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    for i, (f_idx, ev) in enumerate(list(log)[:6]):
        cv2.putText(img, f"f{f_idx}: {ev.type} id={ev.track_id}", (8, 52 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0) if ev.type == "IN" else (80, 80, 255), 1)
    return img


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--source", help="camera index (0) or video path; overrides config")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    src = str(args.source if args.source is not None else cfg["source"])
    src = int(src) if src.isdigit() else src
    is_file = not isinstance(src, int)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {src}")
    if not is_file:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    tracker = PersonTracker(cfg)
    counter = GateCounter(GateConfig.from_dict(cfg["gate"]))
    scale = float(cfg.get("display_scale", 1.0))
    editor = GateEditor(scale)

    win = "Gate IN/OUT"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, editor.on_mouse)

    frame, dets, frame_idx = None, [], 0
    flashes, log = [], deque(maxlen=20)
    paused, show_trails, show_debug = False, True, True

    while True:
        t0 = time.time()
        if not paused and not editor.active:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            dets = tracker(frame)
            for ev in counter.update(frame_idx, dets):
                print(f"[frame {ev.frame}] {ev.type} track_id={ev.track_id}")
                flashes.append((frame_idx, ev))
                log.appendleft((frame_idx, ev))

        if editor.result:                      # new gate drawn
            poly, arrow = editor.result
            counter.set_config(GateConfig.from_dict({**counter.cfg.to_dict(),
                                                     "polygon": poly, "in_direction": arrow}))
            counter.reset()
            editor.result = None

        img = draw_all(frame, counter, dets, flashes, log, scale, show_trails, show_debug, frame_idx)
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
        elif k == ord("g") and frame is not None:
            editor.start_fresh()
        elif k == ord("s"):
            cfg["gate"] = counter.cfg.to_dict()
            with open(args.config, "w") as f:
                json.dump(cfg, f, indent=2)
            print("Saved gate to", args.config)
        elif k == ord("r"):
            counter.reset(); flashes.clear(); log.clear()
        elif k == ord("t"):
            show_trails = not show_trails
        elif k == ord("d"):
            show_debug = not show_debug

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final: IN={counter.count_in} OUT={counter.count_out}")


if __name__ == "__main__":
    main()
