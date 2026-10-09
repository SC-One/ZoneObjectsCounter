"""Synthetic sanity tests for gate_counter.py:  python test_gate_counter.py"""
import random
from gate_counter import CounterConfig, GateCounter, LineGate, RegionConfig, Zone


def box(x, y, h=200):                      # feet at (x, y)
    return (x - h * 0.2, y - h, x + h * 0.2, y)


def run(counter, paths, frames=200, start=1):
    """paths: {id: [(x,y) per frame or None]}"""
    evs = []
    for f in range(frames):
        dets = [(i, box(*p[f])) for i, p in paths.items() if f < len(p) and p[f] is not None]
        evs += counter.update(start + f, dets)
    return evs


def walk(a, b, n):
    return [(a[0] + (b[0] - a[0]) * i / (n - 1), a[1] + (b[1] - a[1]) * i / (n - 1)) for i in range(n)]


def jitter(path, s=2.0):
    r = random.Random(1)
    return [None if p is None else (p[0] + r.uniform(-s, s), p[1] + r.uniform(-s, s)) for p in path]


def types(evs):
    return [(e.type, e.track_id) for e in evs]


# ---- line gate: arbitrary orientation (diagonal line), in_side=left
def test_line():
    cfg = CounterConfig(lines=[LineGate("L", (100, 100), (500, 500), "left")])
    # screen-left of p0->p1 (down-right diagonal) is the upper-right half
    c = GateCounter(cfg)
    ev = run(c, {1: jitter(walk((500, 150), (150, 500 - 150 + 0), 60)),   # from upper-right to lower-left
                 2: jitter(walk((150, 450), (450, 150), 60))})            # lower-left to upper-right
    print("line", types(ev))
    assert ("OUT", 1) in types(ev) and ("IN", 2) in types(ev) and len(ev) == 2


def test_line_jitter_no_count():
    cfg = CounterConfig(lines=[LineGate("L", (0, 300), (640, 300), "left")])
    c = GateCounter(cfg)
    p = [(300 + random.Random(i).uniform(-3, 3), 300 + random.Random(i + 9).uniform(-3, 3)) for i in range(100)]
    assert run(c, {1: p}) == []


def test_line_around_end():
    cfg = CounterConfig(lines=[LineGate("L", (100, 300), (200, 300), "left")])
    c = GateCounter(cfg)
    assert run(c, {1: walk((600, 100), (600, 500), 60)}) == []


# ---- portal, camera INSIDE the room: door zone leads to outside
def portal_cfg(leads):
    return CounterConfig(region=RegionConfig(zones=[Zone("door", "portal",
                         [(250, 100), (390, 100), (390, 250), (250, 250)], leads)]))


def test_portal_camera_inside():
    c = GateCounter(portal_cfg("outside"))
    leave = walk((320, 450), (320, 200), 50)                    # walk to the door ... then vanish
    enter = [None] * 10 + walk((320, 200), (320, 450), 50)      # appear at door, walk into the room
    ev = run(c, {1: leave, 2: enter}, frames=120)
    print("portal inside-cam", types(ev))
    assert ("OUT", 1) in types(ev) and ("IN", 2) in types(ev) and len(ev) == 2


def test_portal_camera_outside():
    c = GateCounter(portal_cfg("inside"))
    enter = walk((320, 450), (320, 200), 50)                    # walk into the door and vanish => IN
    leave = [None] * 10 + walk((320, 200), (320, 450), 50)      # emerge from the door => OUT
    ev = run(c, {1: enter, 2: leave}, frames=120)
    print("portal outside-cam", types(ev))
    assert ("IN", 1) in types(ev) and ("OUT", 2) in types(ev) and len(ev) == 2


def test_portal_loiter_no_count():
    c = GateCounter(portal_cfg("outside"))
    p = walk((320, 300), (320, 240), 20) + [(320, 240)] * 5 + walk((320, 240), (320, 300), 20)
    assert run(c, {1: p}, frames=80) == []                      # walked to the door edge and back


def test_portal_vanish_while_standing_no_count():
    c = GateCounter(portal_cfg("outside"))
    p = [(320, 240)] * 40                                       # standing in portal, then lost
    assert run(c, {1: p}, frames=100) == []


# ---- outside visible zone + default inside
def test_outside_zone():
    cfg = CounterConfig(region=RegionConfig(zones=[Zone("street", "outside",
                        [(0, 0), (640, 0), (640, 200), (0, 200)])]))
    c = GateCounter(cfg)
    ev = run(c, {1: walk((320, 100), (320, 450), 60), 2: walk((320, 450), (320, 20), 60)})
    print("outside zone", types(ev))
    assert ("IN", 1) in types(ev) and ("OUT", 2) in types(ev) and len(ev) == 2


# ---- same track going out and coming back is counted twice, correctly
def test_out_then_in_same_id():
    c = GateCounter(portal_cfg("outside"))
    p = walk((320, 450), (320, 200), 50) + [None] * 20 + walk((320, 200), (320, 450), 50)
    ev = run(c, {1: p}, frames=140)
    print("out then in", types(ev))
    assert types(ev) == [("OUT", 1), ("IN", 1)]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
