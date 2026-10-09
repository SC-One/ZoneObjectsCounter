# Features summary

## Core module: `gate_counter.py`
- Pure Python, standard library only. No YOLO / ByteTrack / OpenCV / numpy inside, so it is easy to port to C++.
- Input per frame: `[(track_id, (x1, y1, x2, y2)), ...]` in pixels. Output: `GateEvent(type "IN"/"OUT", track_id, frame, point, gate_id)`.
- Works with any detector + tracker that gives IDs and boxes.
- Every `track_id` has its own independent state; the same crossing is never counted twice for one track.
- Totals (`count_in`, `count_out`) and counts per gate (`counts[gate_id]`).
- No global direction vector, so it does not depend on camera angle, perspective, or a vertical/horizontal gate.
- Config is plain dataclasses: `CounterConfig`, `LineGate`, `Zone`, `RegionConfig`, with `from_dict()` / `to_dict()`.
- Smoothing of the tracked point (`smooth_window`), track timeout, trail history length.

## Gate types (use any mix, any number, per camera)
### Line gate
- Virtual line on the floor, defined by two points plus the INSIDE side (`in_side`: left/right of p0 to p1).
- Side flip = crossing; flipping toward the inside = IN, otherwise OUT.
- Dead-band against jitter: `max(margin_px, margin_bbox_ratio x bbox height)` (scales with perspective).
- Crossing point must be within the line's extent (`extend`), so walking around the end of the line is ignored.
- Going back and forth counts every real crossing.

### Portal (commit zone)
- Polygon where people appear or vanish (door seen head-on, corridor mouth), with `leads_to` = what is hidden behind it (`outside` / `inside`).
- Walk into the portal and vanish = went to the hidden side. No speed or time condition (slow or long stays still count).
- `commit_lost_frames` is only a debounce for 1-2 frame detector dropouts.
- First seen inside the portal and walking away = came from the hidden side.
- Optional `far_percent`: a person who is smaller than N% of the portal rectangle (`far_metric`: `area` or `height`) counts as already beyond the gate, even while still visible.
- `portal_use_box_center`: also accept the box center inside the portal (partial / occluded bodies).
- Works for a camera inside the room or outside (it only needs `leads_to`).

### Inside / Outside zones
- Optional visible areas; moving from one to the other (after `min_travel_*` movement) = IN or OUT.
- Everything unmarked has the meaning `default_side` (`auto` picks the opposite of what the portal leads to, or of the only zone kind drawn).
- `min_dwell_frames` removes flicker at zone borders.

## Reference point per gate
- Each line / zone / portal can test its own point of the person's box via `ref_point`: `bottom_center`, `center`, `top_center`, `left_center`, `right_center`, `bottom_left`, `bottom_right`, `top_left`, `top_right` (aliases `bottom`, `top`, `left`, `right`).
- If not set, the global `ref_point` (default `bottom_center`, the feet) is used.

## Application: `main.py`
- Input: webcam index, video file, or RTSP/HTTP stream (`--source`).
- YOLO (ultralytics) person detection + ByteTrack tracking, isolated in one small class.
- Real-time GUI: boxes, track IDs, trails, gates, IN/OUT flash on the event position, counters and event log (with gate id).
- Per person: current side `[inside]/[outside]`, live size percentage inside portals, region-colored dot, rings for gates with their own ref point.
- Draw gates any time with the mouse (video freezes on the current frame): line (`l`), portal (`p`), outside zone (`o`), inside zone (`i`), undo (`x`), clear (`c`).
- Adjust while running: `far_percent` (`[` `]`), far measure (`m`), unmarked-area meaning (`f`), per-gate ref point (`y`).
- Fullscreen toggle (`w`) or `"fullscreen": true`; resizable window; pause, trails, debug labels, help line, reset.
- Save the drawn gates with `s`.

## Configuration
- One `config.json` per camera (`--config`), so many cameras / rooms = many configs and processes.
- Gate geometry is stored normalized (0..1), so it survives resolution changes.
- Detector options: model, tracker, `conf`, `iou`, `imgsz`, `device`, display scale, real-time playback for files.
- Old configs still load (unknown keys are ignored, new keys have defaults).

## Quality checks
- `test_gate_counter.py`: synthetic tests for lines at any angle, jitter, walking around a line end, portals from both camera sides, slow / standing / lingering vanishes, far percent (area and height, with return), outside zone, same ID going out then in, and per-gate ref points.
- Not yet validated on real footage; tune thresholds on your own videos.

## Files
| File | Purpose |
|---|---|
| `gate_counter.py` | The independent IN/OUT module |
| `main.py` | GUI application (camera / video, YOLO + ByteTrack) |
| `config.json` | Settings for one camera |
| `README.md` | Full guide: concepts, GUI, config reference, tuning, limits |
| `FEATURES.md` | This summary |
| `test_gate_counter.py` | Synthetic tests of the module |
| `requirements.txt` | `ultralytics`, `opencv-python`, `numpy` |
