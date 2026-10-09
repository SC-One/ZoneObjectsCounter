# ZoneObjectCounter (C++17 / OpenCV)

Counts objects (people) going **IN** and **OUT** from tracker output. It is pure logic: it takes tracked boxes per frame and returns events.
It does not read video, run a detector, or draw anything. Only `opencv_core` is needed (for `cv::Point2f` / `cv::Rect2f`).

## 1. Add to your project

Copy `ZoneObjectCounter.h` and `ZoneObjectCounter.cpp` into your sources.

```cmake
find_package(OpenCV REQUIRED COMPONENTS core)
add_library(zone_counter ZoneObjectCounter.cpp)
target_compile_features(zone_counter PUBLIC cxx_std_17)
target_include_directories(zone_counter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(zone_counter PUBLIC ${OpenCV_LIBS})
```

Everything is in `namespace zoc`. The snippets below assume:

```cpp
#include "ZoneObjectCounter.h"
using namespace zoc;
```

## 2. Basic loop

Every frame, give the counter the objects your tracker currently reports (id + box). Boxes are `cv::Rect2f` in pixels.

```cpp
ZoneObjectCounter counter;
LineGateConfig door;
door.id = "door";
door.p0 = {100.f, 400.f};
door.p1 = {540.f, 400.f};
door.inSide = LineSide::Left;
counter.addLine(door);

for (;;) {
    std::vector<TrackObservation> observations = detectionsForThisFrame();
    for (const ZoneEvent& e : counter.update(frameIndex(), observations)) {
        std::cout << (e.type == EventType::In ? "IN" : "OUT")
                  << " track=" << e.trackId << " gate=" << e.gateId << "\n";
    }
}
```

`TrackObservation` is `{int64_t trackId, int64_t frame, cv::Rect2f box}`. In `update()` the `frame` argument is used; the observation's own `frame` field is only used by `updateFromHistories()`.
Totals: `counter.countIn()`, `counter.countOut()`, per gate: `counter.gateCounts()["door"].in`.

## 3. The three kinds of gate

| Gate | What you define | How it decides |
|---|---|---|
| **Line** (`LineGateConfig`) | Two points on the floor + which side is INSIDE | The tracked point changes side of the line |
| **Portal** (`ZoneConfig`, `ZoneKind::Portal`) | Polygon where people appear / vanish + `leadsTo` (what is hidden behind it) | Vanishes in it: went to the hidden side. Appears in it and walks away: came from the hidden side |
| **Inside / Outside zone** (`ZoneConfig`) | Polygon of a visible area | Moves from one area to the other |

You can mix any number of them in one counter. Use **one gate per physical door**, otherwise one crossing can be reported twice.
No direction vector is used, so camera angle and perspective don't matter.

Helper to get `LineSide` from a point you know is inside the room (left/right is as seen on the image, looking from `p0` to `p1`):

```cpp
LineSide sideOfInsidePoint(cv::Point2f p0, cv::Point2f p1, cv::Point2f insidePoint) {
    const float s = (insidePoint.x - p0.x) * (p1.y - p0.y) - (insidePoint.y - p0.y) * (p1.x - p0.x);
    return s > 0.f ? LineSide::Left : LineSide::Right;
}
```

## 4. Use cases

### 4.1 Corridor, camera looking along it
Put a line across the corridor on the floor. The INSIDE side is the side toward the room.

```cpp
ZoneObjectCounter counter;
LineGateConfig line;
line.id = "corridor";
line.p0 = {150.f, 420.f};
line.p1 = {520.f, 420.f};
line.inSide = sideOfInsidePoint(line.p0, line.p1, {330.f, 200.f});
counter.addLine(line);
```

### 4.2 Camera beside the door (door seen edge-on)
Line along the door threshold on the floor. Both sides of the passage are visible.

```cpp
ZoneObjectCounter counter;
LineGateConfig line;
line.id = "threshold";
line.p0 = {410.f, 120.f};
line.p1 = {470.f, 380.f};
line.inSide = LineSide::Right;
counter.addLine(line);
```

### 4.3 Camera inside the room, facing the door (people vanish into it)
Portal polygon around the door, `leadsTo = Outside`. Walking into it and disappearing counts as OUT, appearing there and walking into the room counts as IN. No time or speed condition: slow or long stays still count when the person finally vanishes.

```cpp
ZoneObjectCounter counter;
ZoneConfig portal;
portal.id = "main_door";
portal.kind = ZoneKind::Portal;
portal.polygon = {{250.f, 100.f}, {390.f, 100.f}, {390.f, 250.f}, {250.f, 250.f}};
portal.leadsTo = Side::Outside;
counter.addZone(portal);
```

### 4.4 Camera outside, facing the door
Same portal, but the hidden side is the inside of the building. Walking into the door = IN.

```cpp
ZoneObjectCounter counter;
ZoneConfig portal;
portal.id = "main_door";
portal.kind = ZoneKind::Portal;
portal.polygon = {{250.f, 100.f}, {390.f, 100.f}, {390.f, 250.f}, {250.f, 250.f}};
portal.leadsTo = Side::Inside;
counter.addZone(portal);
```

### 4.5 People stay visible beyond the gate (open door, glass)
With a camera facing the gate, someone walking away gets smaller. `farPercent` says: inside the portal, a person whose box is smaller than N% of the portal rectangle is already on the hidden side. No vanishing needed.
`farMetric` is `Area` (person box area / portal bounding-rectangle area, default) or `Height`. `0` turns it off. Calibrate it with `trackInfo(id)->sizeRatio` (see 4.12).

```cpp
ZoneObjectCounter counter;
ZoneConfig portal;
portal.id = "glass_door";
portal.kind = ZoneKind::Portal;
portal.polygon = {{250.f, 100.f}, {390.f, 100.f}, {390.f, 250.f}, {250.f, 250.f}};
portal.leadsTo = Side::Outside;
portal.farPercent = 25.f;
portal.farMetric = FarMetric::Area;
counter.addZone(portal);
```

### 4.6 You can see the outside
Mark the outside area as a zone. Everything unmarked is inside (`DefaultSide::Auto` picks the opposite of the only zone kind drawn).
Moving from the unmarked area into the outside zone = OUT, the other way = IN (after `minTravelPx` / `minTravelBBoxRatio` of movement).

```cpp
ZoneObjectCounter counter;
ZoneConfig street;
street.id = "street";
street.kind = ZoneKind::Outside;
street.polygon = {{0.f, 0.f}, {640.f, 0.f}, {640.f, 200.f}, {0.f, 200.f}};
counter.addZone(street);
```

If only the inside is easy to draw, add a `ZoneKind::Inside` polygon instead; the rest of the frame becomes outside.

### 4.7 Several entrances in one room
Add as many gates as you need. Each has its own `id`, and you get totals and per-gate numbers.

```cpp
ZoneObjectCounter counter;

LineGateConfig front;
front.id = "front";
front.p0 = {100.f, 400.f};
front.p1 = {300.f, 400.f};
front.inSide = LineSide::Left;
counter.addLine(front);

ZoneConfig back;
back.id = "back_door";
back.kind = ZoneKind::Portal;
back.polygon = {{450.f, 100.f}, {560.f, 100.f}, {560.f, 240.f}, {450.f, 240.f}};
back.leadsTo = Side::Outside;
counter.addZone(back);

for (const auto& [gate, c] : counter.gateCounts()) {
    std::cout << gate << " in=" << c.in << " out=" << c.out << "\n";
}
```

### 4.8 Your pipeline gives a history of the latest tracked objects
If your tracker hands you, every time, the recent history of each track, convert it to `TrackHistory` and call `updateFromHistories`. Frames already processed are skipped automatically, so you can pass the same rolling window every time.
Pass `currentFrame` so tracks that disappeared are aged out even when no object was detected in the newest frame.

```cpp
ZoneObjectCounter counter;
std::vector<TrackHistory> histories;
for (const MyTrack& t : myTracker.activeTracks()) {
    TrackHistory h;
    h.trackId = t.id;
    for (const MyBox& b : t.recentBoxes) {
        h.observations.push_back({t.id, b.frameNumber, cv::Rect2f(b.x, b.y, b.w, b.h)});
    }
    histories.push_back(std::move(h));
}
std::vector<ZoneEvent> events = counter.updateFromHistories(histories, currentFrameNumber);
```

`MyTrack`, `MyBox` and `myTracker` above stand for your own types.
`updateFromHistories` and `update` can't be mixed on the same instance for the same frames; pick one.

### 4.9 Change the setup at runtime
Everything can be changed while running. Tracks are kept and counts are not reset.

```cpp
counter.addLine(newLine);
counter.removeLine("front");
counter.addZone(newPortal);
counter.removeZone("back_door");

CounterConfig config = counter.config();
config.smoothWindow = 7;
config.region.commitLostFrames = 6;
config.refPoint = RefPoint::Center;
counter.setConfig(config);

counter.resetCounts();
counter.reset();
counter.clearGates();
```

- `addLine` / `addZone` return `false` if the `id` is empty or already exists (a zone also needs at least 3 points). `removeLine` / `removeZone` return `false` if the id doesn't exist.
- Changing zones restarts every track's inside/outside knowledge (a person already inside is re-learned from where they are next seen). Adding or removing a line only affects that line.
- If the camera moves or zooms, replace the gates (`clearGates()` then add new ones) or call `setConfig`.
- `resetCounts()` zeroes counters. `reset()` also forgets all tracks.

### 4.10 Different reference point per gate
Each gate tests one point of the person's box: `refPoint` on the gate, or the global `CounterConfig::refPoint` (default `BottomCenter`, the feet).
Values: `BottomCenter`, `Center`, `TopCenter`, `LeftCenter`, `RightCenter`, `BottomLeft`, `BottomRight`, `TopLeft`, `TopRight`.

```cpp
LineGateConfig threshold;
threshold.id = "threshold";
threshold.p0 = {100.f, 400.f};
threshold.p1 = {300.f, 400.f};
threshold.refPoint = RefPoint::BottomCenter;
counter.addLine(threshold);

ZoneConfig upperDoor;
upperDoor.id = "upper_door";
upperDoor.kind = ZoneKind::Portal;
upperDoor.polygon = {{450.f, 100.f}, {560.f, 100.f}, {560.f, 240.f}, {450.f, 240.f}};
upperDoor.refPoint = RefPoint::Center;
counter.addZone(upperDoor);
```

Use the feet for floor lines and thresholds. Use `Center` / `TopCenter` when feet are often hidden or cut by the frame. Use `LeftCenter` / `RightCenter` when people walk past a wall sideways.

### 4.11 One instance per camera
Each camera needs its own `ZoneObjectCounter` (own gates, own track IDs, own counts).

```cpp
std::map<std::string, ZoneObjectCounter> counters;
counters.emplace("entrance", ZoneObjectCounter(entranceConfig));
counters.emplace("office", ZoneObjectCounter(officeConfig));

counters.at("entrance").update(frameNumberA, observationsA);
counters.at("office").update(frameNumberB, observationsB);
```

The class is not thread-safe. Use one thread per instance, or protect an instance with a mutex.

### 4.12 Debugging / visualization with OpenCV
The counter exposes what it knows about each track so you can draw it yourself.

```cpp
void drawDebug(cv::Mat& image, const ZoneObjectCounter& counter) {
    for (const auto& zone : counter.config().region.zones) {
        std::vector<cv::Point> poly(zone.polygon.begin(), zone.polygon.end());
        cv::polylines(image, poly, true, cv::Scalar(0, 165, 255), 2);
    }
    for (const auto& line : counter.config().lines) {
        cv::line(image, line.p0, line.p1, cv::Scalar(255, 0, 255), 2);
    }
    for (int64_t id : counter.trackIds()) {
        std::optional<TrackInfo> info = counter.trackInfo(id);
        if (!info || info->trail.empty()) continue;
        std::vector<cv::Point> trail(info->trail.begin(), info->trail.end());
        cv::polylines(image, trail, false, cv::Scalar(0, 255, 0), 2);
        std::string label = "ID " + std::to_string(id);
        if (info->side) label += *info->side == Side::Inside ? " [inside]" : " [outside]";
        if (info->sizeRatio) label += " " + std::to_string(int(*info->sizeRatio * 100)) + "%";
        cv::putText(image, label, trail.back(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}
```

`TrackInfo`: `trail` (smoothed points), `side` (inside/outside, empty if unknown), `region` (last region seen: inside / outside / portal), `sizeRatio` (person size / portal rectangle size, only while inside a portal, for `farPercent` calibration), `lastEvent`, `lastFrame`.
`counter.defaultSide()` tells what the unmarked area currently means.

### 4.13 Gates stored as normalized coordinates in your own config
The class works in pixels of the frames you give it. If you store gates as 0..1 fractions, convert when you build them:

```cpp
cv::Point2f toPixels(cv::Point2f normalized, cv::Size frameSize) {
    return {normalized.x * frameSize.width, normalized.y * frameSize.height};
}

LineGateConfig makeLine(const std::string& id, cv::Point2f n0, cv::Point2f n1, LineSide side, cv::Size size) {
    LineGateConfig line;
    line.id = id;
    line.p0 = toPixels(n0, size);
    line.p1 = toPixels(n1, size);
    line.inSide = side;
    return line;
}
```

## 5. Rules to follow

- Gates and boxes must be in the **same coordinate space** (for example full-resolution frame coordinates, not detector-input coordinates).
- Frame numbers must **increase** for every `update`. Frame-based settings (`minDwellFrames`, `commitLostFrames`, `trackTimeoutFrames`) are measured in frame-number differences. If you process only every Nth frame and pass the real frame numbers, scale these settings; or pass a counter that increases by 1 per processed frame.
- Call `update` for **every** processed frame, even with an empty list, so disappeared tracks are detected.
- Track IDs must be stable per person. If your tracker changes the ID of a person right at the gate, that crossing can be missed.
- Duplicate IDs within one frame, or a frame number not greater than the track's last one, are ignored.

## 6. Settings reference

`CounterConfig`

| Field | Default | Meaning |
|---|---|---|
| `refPoint` | `BottomCenter` | Box point used by gates without their own `refPoint` |
| `smoothWindow` | 5 | Moving average length (frames) over the tracked point |
| `trackTimeoutFrames` | 30 | Forget a track unseen for this many frames |
| `maxHistory` | 300 | Trail length kept per track |
| `lines` | empty | Line gates |
| `region` | see below | Zones and region settings |

`LineGateConfig`: `id`, `p0`, `p1`, `inSide` (`Left`/`Right` of `p0 -> p1` as seen on the image), `marginPx` 8, `marginBBoxRatio` 0.08 (dead-band `max(px, ratio * box height)`), `extend` 0.15 (crossing accepted this fraction beyond the segment ends), `refPoint` (optional).

`ZoneConfig`: `id`, `kind` (`Portal`/`Inside`/`Outside`), `polygon`, `leadsTo` (portal only: `Outside`/`Inside`), `farPercent` 0 (off), `farMetric` `Area`/`Height`, `refPoint` (optional).

`RegionConfig`: `zones`, `defaultSide` (`Auto`/`Inside`/`Outside`, meaning of the unmarked area), `minDwellFrames` 3 (frames in a region before it is trusted), `commitLostFrames` 8 (lost this long while last seen in a portal = vanished; only a debounce for detector dropouts), `portalUseBoxCenter` true (a person also counts as in the portal when the box center is inside, helps with partial bodies), `minTravelPx` 30 and `minTravelBBoxRatio` 0.35 (minimum movement for a zone-to-zone change: `max(px, ratio * box height)`).

## 7. Tuning

| Problem | Try |
|---|---|
| Line misses crossings | Lower `marginPx` / `marginBBoxRatio` (not below the feet jitter), make sure the tracked point is visible (`Center` if feet are hidden), lengthen the line or raise `extend` |
| Line gives extra counts | Raise `marginBBoxRatio`; people walking past the line end: lower `extend` |
| Portal misses people who fade out | The last position must be inside the polygon: enlarge the portal, keep `portalUseBoxCenter`, lower `commitLostFrames`; or set `farPercent` |
| Portal gives false counts | Shrink the polygon tightly around the door; raise `commitLostFrames`; if `farPercent` fires too early, lower it |
| IN and OUT swapped | Line: flip `inSide`; portal: flip `leadsTo`; zones: set `defaultSide` explicitly |
| Flicker at zone borders | Raise `minDwellFrames` |
| Jittery tracks | Raise `smoothWindow` |
| People lost while crossing | Raise `trackTimeoutFrames` |

## 8. Limits

- Quality depends on stable track IDs and on the gate being drawn where the tracked point really passes.
- A portal relies on the tracker losing a person when they go through the door and not detecting them inside the door area.
- Groups are counted only if the tracker separates them.
- Tune the settings on your own footage.
