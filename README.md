# Gate IN/OUT People Counter

Counts people **entering (IN)** and **leaving (OUT)** through a gate, from a camera or a video file.
It has a debug GUI so you can *see* what the algorithm is doing.

---

## 1. Big picture

```
 camera / video ──► YOLO (finds people) ──► ByteTrack (gives each person an ID)
                                                   │
                                   [(track_id, bbox), ...] per frame
                                                   ▼
                                       GateCounter   (gate_counter.py)
                                       pure logic, no AI libraries
                                                   │
                                          IN / OUT events + counts
                                                   ▼
                                           GUI (main.py) draws everything
```

| File | Role |
|---|---|
| `gate_counter.py` | **The module.** IN/OUT logic only. Standard library only. Knows nothing about YOLO, ByteTrack or OpenCV. |
| `main.py` | The application: opens camera/video, runs YOLO+ByteTrack, feeds `GateCounter`, shows the GUI. |
| `config.json` | All settings for one camera (source, model, gate, thresholds). |
| `CONFIG.md` | Short key-by-key reference of `config.json`. |

---

## 2. Install and run

```bash
pip install -r requirements.txt          # ultralytics, opencv-python, numpy

python main.py --source 0                # webcam 0
python main.py --source video.mp4        # video file
python main.py --config room2.json       # use another config file
```

The first run downloads `yolov8n.pt` automatically.

---

## 3. What you see in the GUI

| What | Meaning |
|---|---|
| **Yellow/orange transparent polygon** | The **Gate** (the zone). It is *your* polygon. |
| **Magenta arrow labeled `IN`** | The **IN direction**. Walking along this arrow = IN. Walking against it = OUT. |
| **Colored box + `ID 3 STATE`** | A tracked person and their ID. Each ID has its own color. |
| **Colored line behind a person** | Their trajectory (trail). |
| **Small dot on the person's feet** | The point used for the logic. **Green** = inside the gate, **red** = outside. |
| **`IN #3` / `OUT #5` large text** | An event, shown for about 1.5 seconds where it happened. |
| **Top-left panel** | Total `IN` / `OUT` counts and the last events. |
| `STATE` in the label | `IDLE` (outside, waiting), `IN_GATE` (inside, being evaluated), `COUNTED` (already counted, locked). |

### "Why is there a rectangle? I want my own polygon!"

That rectangle is **only the default placeholder** from `config.json`
(the 4 points `[0.25,0.3] [0.75,0.3] [0.75,0.8] [0.25,0.8]`).
It is not special. Replace it any time:

1. Press **`g`** (the video freezes on the current frame).
2. Click the points of your polygon on the image (as many as you want, at least 3).
   `Backspace` removes the last point.
3. Press **`Enter`** to close the polygon.
4. Click **two points** for the direction arrow: **start**, then **end**
   (or press `Enter` to keep the existing arrow).
5. Done. Counts reset and the new gate is active.
6. Press **`s`** to save it into the config file so it is used next time.

`Esc` cancels while editing.

### Keys

| Key | Action |
|---|---|
| `g` | Draw a new gate polygon + direction |
| `s` | Save current gate to the config file |
| `r` | Reset counters |
| `space` | Pause / resume |
| `t` | Show/hide trails |
| `d` | Show/hide state text in labels |
| `q` / `Esc` | Quit |

---

## 4. Gate and direction (the most important concept)

### The Gate (polygon)
A zone in the image that people walk **through**. Draw it over the doorway / passage, on the **floor area**
where people's feet pass (the logic follows the *feet*, bottom-center of the box).

Tips:
- Make it large enough that a person is inside it for at least a few frames.
- It can be any shape and any orientation: a tilted rectangle, a trapezoid (perspective), etc.
- Don't make it so large that it also covers people just standing nearby.

### The direction arrow
An arrow from point A to point B that says: **"walking this way means IN."**

- Draw it **across the gate, pointing toward the inside of the room**.
- Only its *direction* matters, not its length or position.
- It does **not** have to be horizontal or vertical. Draw it however people actually walk.
- If your IN/OUT are swapped, just draw the arrow the opposite way.

```
   outside                inside
      ○ ──►  [  GATE  ]  ──►  ○        arrow points to the inside → this way = IN
```

### Coordinates are normalized (0..1)
In `config.json`, points are fractions of the frame: `[0,0]` = top-left, `[1,1]` = bottom-right.
So the same config works if you change the camera resolution. You normally never edit them by hand:
use `g` then `s`.

---

## 5. How the counting logic works

Every person (track ID) has its **own** small state machine:

```
 IDLE ──feet enter gate──► IN_GATE ──moved far enough along the arrow──► COUNTED
   ▲                          │                                             │
   └──── left gate, no ◄──────┘                                             │
         decision (turned back)                    outside the gate for N frames
   ▲                                                                        │
   └────────────────────────────────────────────────────────────────────────┘
```

1. **IDLE → IN_GATE**: the smoothed feet point enters the polygon. The previous point is remembered as the *entry point*.
2. **While IN_GATE**: the displacement `current − entry` is projected onto the arrow.
   - projection > 0 → **IN**, projection < 0 → **OUT**.
   - It is only accepted if (a) the distance is big enough (`min_displacement_*`) and
     (b) the movement is mostly along the arrow (`min_alignment`), so someone walking *sideways* across the gate is ignored.
3. **Leaves the gate** without enough movement → no event (e.g. someone who walked in a bit and turned back).
4. **Track lost while inside** (occlusion, left the view) → one final decision using the last known position.
5. **COUNTED**: the track is locked. It can only count again after it has been **outside** the gate for `rearm_frames_outside` frames. This is what prevents counting the same person many times.

The movement is smoothed (moving average) to remove bounding-box jitter, and the distance threshold scales with the person's box height, so it adapts to perspective (people far away look smaller).

---

## 6. Using the module in your own code

`gate_counter.py` is independent. Copy that one file into any project.

```python
from gate_counter import GateConfig, GateCounter

cfg = GateConfig(
    polygon=[(300, 200), (700, 200), (700, 500), (300, 500)],   # PIXELS here
    in_direction=((500, 250), (500, 450)),                      # PIXELS here
)
counter = GateCounter(cfg)

frame_idx = 0
for frame in your_frames:
    frame_idx += 1
    tracks = [(12, (x1, y1, x2, y2)), (15, (x1, y1, x2, y2))]   # from ANY detector+tracker
    for ev in counter.update(frame_idx, tracks):
        print(ev.type, ev.track_id, ev.frame, ev.point)          # "IN" / "OUT"

print(counter.count_in, counter.count_out)
```

Rules:
- The module itself uses **pixels**. The 0..1 normalization is only done by `main.py` when reading/saving the config.
- Call `update()` **once per frame**, with `frame_idx` increasing. Pass only the people visible in that frame.
- You can use any detector/tracker (YOLO, RT-DETR, DeepSORT...) as long as you give `(track_id, (x1,y1,x2,y2))`.
- Useful extras: `counter.tracks` (per-track state + `trail`), `counter.reset()`, `counter.set_config(new_cfg)`.
- No GUI needed: you can run it headless on a server and only read the events.

---

## 7. Using it with different cameras / different rooms

**One camera = one config file = one running process.** Gate and arrow depend on where the camera is, so each camera needs its own.

```
configs/
  entrance.json      # source: rtsp://.../cam1   + gate drawn for that view
  office_door.json   # source: rtsp://.../cam2   + gate drawn for that view
  warehouse.json     # source: video or camera 2
```

Setup for each new camera:

1. Copy `config.json` to a new file, e.g. `office_door.json`, and set `"source"` (camera index, video path or RTSP URL).
2. Run `python main.py --config office_door.json`.
3. Press `g`, draw the gate over the doorway and the arrow toward the inside. Press `s`.
4. Walk through a few times (or play a recording) and watch the events. Tune the parameters (section 8).
5. Run one terminal/process per camera:
   ```bash
   python main.py --config configs/entrance.json
   python main.py --config configs/office_door.json
   ```

Notes:
- If the camera is **moved or rotated**, redraw the gate (`g`, `s`).
- Each camera has its own `GateCounter`, so IDs and counts are **separate per camera**. The same person seen by two cameras is counted by each independently.
- For several cameras in one Python program, create one `GateCounter` per camera and feed each its own tracks.
- Good camera placement: looking at the passage so people walk **across the arrow direction clearly** (towards/away from the camera or side-on both work), with the feet visible inside the gate.

---

## 8. Tuning (in the `gate` section of the config)

| Problem | Try |
|---|---|
| Real entries are missed | Lower `min_displacement_px` / `min_displacement_bbox_ratio`; enlarge the polygon; lower `min_alignment` |
| False counts (people just standing/passing by) | Raise `min_displacement_*`; shrink the polygon; raise `min_alignment` |
| Walking sideways across the gate gets counted | Raise `min_alignment` (e.g. 0.6) |
| Same person counted twice | Raise `rearm_frames_outside`; improve tracking (bigger model, lower `conf` to keep tracks alive) |
| IN and OUT are swapped | Redraw the arrow the other way |
| Counts jump around due to jitter | Raise `smooth_window` |
| People missed when occluded | Raise `track_timeout_frames` |
| Feet hidden (only upper body visible) | Set `ref_point` to `center` |

Detector-level (top of config): `model` (`yolov8n.pt` fast, `yolov8s/m.pt` more accurate), `conf`, `imgsz`, `device` (`"cpu"`, `"0"` for GPU).
The full parameter list is in `CONFIG.md`.

---

## 9. Troubleshooting

- **Nothing is counted**: check that the green/red dot on a person turns green inside the polygon, the arrow points the way people walk, and the label reaches `COUNTED`. If it stays `IN_GATE` then goes back to `IDLE`, the movement was too small or not aligned: lower the thresholds.
- **IDs keep changing**: the tracker loses people. Use a better model, check lighting, or lower `conf`.
- **Window too large**: set `display_scale` (e.g. `0.6`).
- **Video plays too fast/slow**: `realtime_playback` (files only). Slow machines can't keep real-time; set `imgsz` smaller or use a lighter model.

---

## 10. Porting `gate_counter.py` to C++

The module was written for this: only loops, structs, `deque`, and `dict` of per-track states.
Mapping: `GateConfig` → struct, `TrackState` → struct, `Dict` → `std::unordered_map<int, TrackState>`, `deque` → `std::deque`,
`point_in_polygon` and the median/projection math are plain arithmetic. No external libraries are needed.
