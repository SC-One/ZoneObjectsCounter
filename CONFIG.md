# config.json reference

## Top level
| key | meaning |
|---|---|
| `source` | camera index (`"0"`) or video path. CLI `--source` overrides it |
| `model` | YOLO weights (`yolov8n.pt`, ...) |
| `tracker` | ultralytics tracker yaml (`bytetrack.yaml`) |
| `conf`, `iou`, `imgsz`, `device` | detector params (`device`: `null`, `"cpu"`, `"0"`) |
| `display_scale` | GUI window scale (e.g. `0.6` for big frames) |
| `realtime_playback` | video files only: play at native FPS |

## `gate` (geometry is NORMALIZED 0..1, independent of resolution)
| key | meaning |
|---|---|
| `polygon` | gate/ROI points `[[x,y],...]`, x/y in 0..1 (fraction of frame width/height), >= 3 points, any shape/orientation |
| `in_direction` | arrow `[[x0,y0],[x1,y1]]` (0..1). Moving along it = **IN**, opposite = **OUT** |

Press `g` in the GUI to redraw both, `s` to save (written normalized).

## `gate` (logic params, pixels where noted)
| key | default | meaning |
|---|---|---|
| `ref_point` | `bottom_center` | point tracked on the bbox: `bottom_center` (feet) or `center` |
| `smooth_window` | 5 | moving-average length (frames) on the tracked point |
| `min_displacement_px` | 30 | min travel along the IN arrow to count (pixels) |
| `min_displacement_bbox_ratio` | 0.5 | ...or this x median bbox height; the larger of the two is used (scales with perspective) |
| `min_alignment` | 0.4 | min `|travel along arrow| / |total travel|`; rejects people walking sideways across the gate |
| `min_frames_in_gate` | 3 | frames inside the polygon before an early decision |
| `rearm_frames_outside` | 10 | frames outside the gate before a counted track may count again |
| `track_timeout_frames` | 30 | unseen frames before a track is dropped (final decision if it was mid-crossing) |
| `max_history` | 300 | trail length kept per track |

## GUI gate editor (`g`, any time)
1. Video freezes on the current frame; the current gate stays visible.
2. Left-click polygon points. `Backspace` undoes, `Enter` finishes.
3. Click 2 points for the IN arrow (start -> end), or press `Enter` to keep the current arrow.
4. `Esc` cancels. Counts are reset when a new gate is applied.
5. `s` saves to the config file.
