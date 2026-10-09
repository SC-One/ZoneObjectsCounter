# People IN/OUT Counter (any camera angle)

Counts people **entering (IN)** and **leaving (OUT)** from a camera or video file, with a debug GUI.
v2: the old "one polygon + one direction arrow" design was replaced, because a single image-space
direction does not hold for corridors, perspective, door-facing cameras or people who vanish into a door.

## 1. How it works now

```
camera/video ─► YOLO + ByteTrack ─► [(track_id, bbox)] ─► GateCounter (gate_counter.py, pure Python) ─► IN/OUT events
                (main.py only)                                       ▲
                                                 gates drawn in the GUI: lines, portals, zones
```

The counter never uses a global direction. It uses **topology**: which *side* of a line a person is on, or
which *region* (inside / outside / portal) they came from and went to. That is independent of camera angle.

| Gate type | Idea | Use when |
|---|---|---|
| **Line** | A virtual line on the floor. Crossing it flips the person's side. You also mark which side is INSIDE. | You can see both sides of the passage. |
| **Portal** (commit zone) | A polygon where people **appear / vanish** (a door seen head-on, a corridor mouth). You say what is hidden behind it: `outside` or `inside`. Walk in and vanish = went to the hidden side. Appear there and walk away = came from the hidden side. | You cannot see the other side. |
| **Inside / Outside zone** | Optional visible areas. Moving from one to the other = IN or OUT. Anything unmarked = `default_side`. | You can see outside (glass door, lobby, street). |

You can combine any number of lines and zones in one camera (a room with several entrances).
Use **one gate per physical door**, otherwise one real crossing can be reported twice.

## 2. Quick start

```bash
pip install -r requirements.txt            # ultralytics, opencv-python, numpy
python main.py --source 0                  # webcam
python main.py --source video.mp4          # file
python main.py --source rtsp://...         # IP camera
python main.py --config room2.json         # another camera's config
python test_gate_counter.py                # synthetic tests of the logic (no camera needed)
```

First run: no gates exist (the GUI says so). Press a key below to draw one, then `s` to save.

## 3. Which gate for which camera?

| Camera situation | Use |
|---|---|
| **Corridor**, camera looking along it | **Line** across the corridor on the floor, mark the room side as INSIDE |
| **Side-on / orthogonal** to the door (you see people pass the threshold) | **Line** along the door threshold on the floor |
| **Head-on, inside the room** looking at the door (people vanish into / appear from it) | **Portal** around the door, hidden = `outside` |
| **Head-on, outside** looking at the door | **Portal** around the door, hidden = `inside` |
| **Overhead** camera | Line on the threshold (or a portal) |
| You can **see outside** through the door | Line, or an **Outside zone** polygon (+ optional portal) |
| Only **inside** is visible, no clear door | Portal where people appear/disappear |
| **Several entrances** in one view | One line/portal per entrance |

Guidelines: lines are most precise when feet cross them. Portals depend on the tracker losing/finding people at the door,
so draw them tightly around the door, not over the whole room.

## 4. Drawing gates in the GUI (any time)

The video freezes on the current frame while you draw. `Esc` cancels.

| Key | Action |
|---|---|
| `l` | **Line**: click start, click end, then click **one point on the INSIDE side**. An arrow marked "IN side" is drawn. |
| `p` | **Portal**: click polygon points, `Enter`, then `O` (hidden side = outside) or `I` (hidden = inside) |
| `o` | **Outside zone** polygon (`Enter` to finish) |
| `i` | **Inside zone** polygon (`Enter` to finish) |
| `Backspace` | undo last point while drawing |
| `x` / `c` | delete last gate / clear all gates |
| `f` | cycle the meaning of the unmarked area: `auto` → `inside` → `outside` |
| `s` | **save** to the config file |
| `r` | reset counters |
| `space`, `t`, `d`, `h`, `q` | pause, trails, debug labels, help line, quit |

Adding or deleting a gate resets counts and track states.

What you see: magenta lines with an `IN side` arrow; orange = portal, green = inside zone, blue = outside zone (labeled);
colored boxes with `ID n [inside|outside]` (the side the tracker currently believes the person is on);
a dot on the feet (colored by region); `IN #3` / `OUT #5` flashes; top-left counts and event log with the gate id.

## 5. Logic details

**Line gate** (per track, per line): signed distance of the (smoothed) feet point to the line. Inside a dead-band
(`max(margin_px, margin_bbox_ratio * bbox height)`) nothing changes, so jitter can't flip the side. When the side flips,
the crossing point must lie within the segment (± `extend`), otherwise the person just walked around the line's end.
Flip toward the `in_side` = IN, otherwise OUT. A person who goes back and forth is counted each real crossing.

**Region model** (per track): the track has a `side` (`inside`/`outside`), initialised from where it is first confidently seen
(or from the portal's hidden side if it first appears inside a portal). It changes, producing an event, when:
- it leaves a portal into a visible area on the *other* side than the portal's hidden side, after moving enough (`min_travel_*`);
- it moves between an inside and an outside zone after moving enough;
- it **vanishes inside a portal** (lost for `commit_lost_frames`) after really walking toward it (`vanish_lookback_frames`, `min_travel_*`) and its side differs from the portal's hidden side.
A person who walks to the door edge and returns, or stands in the portal and is lost, produces nothing.
Region changes need `min_dwell_frames` frames, which removes boundary flicker.

Each track has its own state, so the same ID is never counted twice for the same crossing.
If the tracker gives a person a new ID right at the gate, that crossing can be missed or split. This is a tracker limit.

## 6. Config reference (`config.json`)

Geometry is **normalized 0..1** (fraction of frame width/height), so resolution changes don't break it. Normally you never edit it by hand.

Top level: `source` (index / path / RTSP), `model`, `tracker`, `conf`, `iou`, `imgsz`, `device` (`null`, `"cpu"`, `"0"`),
`display_scale`, `realtime_playback` (files only).

`counter`:

| key | default | meaning |
|---|---|---|
| `ref_point` | `bottom_center` | point of the bbox that is tracked: feet, or `center` if feet are often hidden |
| `smooth_window` | 5 | moving average (frames) |
| `track_timeout_frames` | 30 | forget a track unseen this long |
| `max_history` | 300 | trail length |
| `lines[]` | `[]` | `{id, p0, p1, in_side, margin_px, margin_bbox_ratio, extend}`; defaults 8 / 0.08 / 0.15; `in_side` = `left`/`right` of p0→p1 as seen on screen |
| `region.zones[]` | `[]` | `{id, kind: portal/inside/outside, polygon, leads_to}`; `leads_to` (`outside`/`inside`) only for portals |
| `region.default_side` | `auto` | meaning of unmarked area. `auto`: opposite of what the portal leads to; or opposite of the only zone kind drawn |
| `region.min_dwell_frames` | 3 | frames in a region before it is trusted |
| `region.commit_lost_frames` | 8 | lost this long inside a portal = vanished into it |
| `region.vanish_lookback_frames` | 30 | window used to check the person was really moving |
| `region.min_travel_px` / `min_travel_bbox_ratio` | 30 / 0.35 | min movement to accept a region change: `max(px, ratio * bbox height)` |

## 7. Using the module in your own code

`gate_counter.py` needs nothing but the standard library. Copy it anywhere. It works in **pixels**.

```python
from gate_counter import CounterConfig, LineGate, Zone, RegionConfig, GateCounter

cfg = CounterConfig(
    lines=[LineGate("door", (100, 400), (500, 400), in_side="left")],
    region=RegionConfig(zones=[Zone("gate2", "portal", [(250,100),(390,100),(390,250),(250,250)], leads_to="outside")]),
)
counter = GateCounter(cfg)
for frame_idx, tracks in enumerate(your_stream, 1):          # tracks = [(id, (x1,y1,x2,y2)), ...]
    for ev in counter.update(frame_idx, tracks):             # call once per frame
        print(ev.type, ev.track_id, ev.gate_id)              # "IN"/"OUT"
print(counter.count_in, counter.count_out, counter.counts)   # counts per gate id
```
Any detector+tracker works. `CounterConfig.from_dict()/to_dict()` load/save the JSON shape above (in pixels).

## 8. Several cameras / rooms

One camera = one config file = one process. Gates depend on where the camera is looking, so draw them per camera.

```bash
python main.py --config configs/entrance.json
python main.py --config configs/office.json
```
For each new camera: copy `config.json`, set `source`, run, draw gates (section 4), press `s`, watch events, tune (section 9).
If the camera moves, redraw. Counts and IDs are per camera.

## 9. Tuning

| Problem | Try |
|---|---|
| Misses on a line | lower `margin_*` (but not below the feet jitter); make sure feet are visible, else `ref_point: center`; lengthen the line or raise `extend` |
| Extra counts on a line | raise `margin_bbox_ratio`; people walking past the line end: lower `extend` |
| Portal misses vanishes | lower `commit_lost_frames`, `min_travel_*`; enlarge the portal slightly |
| Portal false counts | shrink the portal tightly around the door; raise `commit_lost_frames` / `min_travel_*` |
| IN/OUT swapped | line: re-draw with the inside click on the other side; portal: `O`/`I` the other way; zones: press `f` |
| Region flicker | raise `min_dwell_frames` |
| IDs change a lot | bigger model (`yolov8s/m`), lower `conf`, better lighting |

## 10. Limits (read this)

- Tested only with synthetic tracks (`test_gate_counter.py`: lines at any angle, jitter, walking around a line, portals from both camera sides, outside zone, out-then-in with the same ID). **Not yet validated on real footage**, so expect to tune on yours.
- Portal logic relies on the tracker keeping the ID until the person vanishes and on the detector not finding people inside the door area.
- Lines/zones use the feet point on the floor; camera lens distortion, strong occlusion or very tall camera tilt can need a looser margin.
- Groups entering together get an ID each only if the detector/tracker separates them.

## 11. Porting `gate_counter.py` to C++

Only structs, loops, `std::deque`, `std::unordered_map`, and plain arithmetic: `CounterConfig`/`TrackState` → structs,
`point_in_polygon`, signed-distance, median and distance helpers translate directly. No external libraries.
