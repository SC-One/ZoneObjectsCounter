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


def test_portal_stay_then_vanish():
    c = GateCounter(portal_cfg("outside"))
    p = walk((320, 450), (320, 200), 40) + [(320, 200)] * 90      # walk in, stay ~3 s, then lost
    ev = run(c, {1: p}, frames=200)
    print("stay then vanish", types(ev))
    assert types(ev) == [("OUT", 1)]


def test_portal_edge_jitter_then_lost_no_count():
    c = GateCounter(portal_cfg("outside"))
    p = [(320, 300)] * 60 + jitter([(320, 255)] * 10, 6)          # standing at the portal edge, lost
    assert run(c, {1: p}, frames=150) == []


# ---- slow / lingering disappearance must still count (no time or speed condition)
def test_portal_slow_vanish():
    c = GateCounter(portal_cfg("outside"))
    p = walk((320, 450), (320, 240), 40) + walk((320, 240), (320, 232), 90)   # creeps for 3 s, then lost
    ev = run(c, {1: p}, frames=200)
    print("slow vanish", types(ev))
    assert types(ev) == [("OUT", 1)]


def test_portal_stand_then_vanish():
    c = GateCounter(portal_cfg("inside"))
    p = walk((320, 450), (320, 240), 40) + [(320, 240)] * 120               # stands still 4 s, then lost
    assert types(run(c, {1: p}, frames=260)) == [("IN", 1)]


# ---- far_percent: person visible but small inside the portal => hidden side
def far_cfg(leads, pct=50):
    cfg = portal_cfg(leads)
    cfg.region.zones[0].far_percent = pct
    return cfg


def shrink_path(h_end=40):
    a = [(x, y, 200) for x, y in walk((320, 450), (320, 240), 30)]
    b = [(320, 240 - i * 0.5, 200 - (200 - h_end) * i / 39) for i in range(40)]   # walks away, shrinks
    return a + b


def test_far_percent_camera_inside():
    c = GateCounter(far_cfg("outside", 50))                      # area metric: far below 50% of the portal rect
    ev = run(c, {1: shrink_path()}, frames=70)                   # person stays visible to the end
    print("far inside-cam", types(ev))
    assert types(ev) == [("OUT", 1)]


def test_far_percent_height_metric():
    cfg = far_cfg("outside", 50)
    cfg.region.zones[0].far_metric = "height"                    # rect h=150 -> far below 75 px
    c = GateCounter(cfg)
    assert types(run(c, {1: shrink_path()}, frames=70)) == [("OUT", 1)]


def test_far_percent_area_threshold():
    # portal rect 140x150=21000. Person box (0.4h x h) area 0.4h^2: 15% -> far when h < 88
    cfg = far_cfg("outside", 15)
    c = GateCounter(cfg)
    ev = run(c, {1: shrink_path(h_end=120)}, frames=70)         # only shrinks to 120 px -> never "far"
    assert ev == []
    c = GateCounter(far_cfg("outside", 15))
    assert types(run(c, {1: shrink_path(h_end=60)}, frames=70)) == [("OUT", 1)]


def test_far_percent_camera_outside():
    c = GateCounter(far_cfg("inside", 50))
    assert types(run(c, {1: shrink_path()}, frames=70)) == [("IN", 1)]


def test_far_percent_off_no_event():
    c = GateCounter(far_cfg("outside", 0))
    assert run(c, {1: shrink_path()}, frames=70) == []           # still visible, feature off


def test_far_then_return():
    c = GateCounter(far_cfg("outside", 50))
    back = [(320, 240 - 19.5 + i * 0.5, 40 + 160 * i / 39) for i in range(40)]
    p = shrink_path() + back + [(x, y, 200) for x, y in walk((320, 240), (320, 450), 40)]
    ev = run(c, {1: p}, frames=160)
    print("far then return", types(ev))
    assert types(ev) == [("OUT", 1), ("IN", 1)]


# ---- per-gate reference point
def test_per_line_ref_point():
    # person (h=200) walks up through a horizontal line at y=300: the box CENTER crosses ~100 px before the feet
    def first_event_frame(ref):
        c = GateCounter(CounterConfig(lines=[LineGate("L", (0, 300), (640, 300), "left", ref_point=ref)]))
        ev = run(c, {1: walk((320, 700), (320, 100), 120)})
        assert len(ev) == 1, ev
        return ev[0].frame
    f_feet, f_center, f_top = first_event_frame(None), first_event_frame("center"), first_event_frame("top_center")
    print("ref frames feet/center/top", f_feet, f_center, f_top)
    assert f_top < f_center < f_feet          # walking up: the higher bbox point reaches the line first


def test_per_zone_ref_point():
    def cfg(ref):
        z = Zone("door", "portal", [(250, 100), (390, 100), (390, 250), (250, 250)], "outside", ref_point=ref)
        return CounterConfig(region=RegionConfig(zones=[z], portal_use_box_center=False))
    p = walk((320, 500), (320, 300), 50)             # feet stop at y=300 (below the portal), center reaches y=200 (inside)
    assert run(GateCounter(cfg(None)), {1: p}, frames=120) == []                      # feet never enter
    assert types(run(GateCounter(cfg("center")), {1: p}, frames=120)) == [("OUT", 1)]  # center does


def test_ref_point_names():
    from gate_counter import reference_point
    b = (100, 100, 140, 300)
    assert reference_point(b, "bottom_center") == (120, 300) and reference_point(b, "left") == (100, 200)
    assert reference_point(b, "top_right") == (140, 100)
    try:
        reference_point(b, "nope"); assert False
    except ValueError:
        pass


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
