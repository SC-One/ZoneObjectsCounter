#include "ZoneObjectCounter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <set>

namespace zoc {

namespace {

constexpr int kRefCount = 9;
constexpr size_t kHeightWindow = 30;
constexpr int64_t kNoFrame = std::numeric_limits<int64_t>::min();

using ActiveRefs = std::array<bool, kRefCount>;

int refIndex(RefPoint ref) { return static_cast<int>(ref); }

cv::Point2f referencePoint(const cv::Rect2f& box, RefPoint ref) {
    static const float kFraction[kRefCount][2] = {
        {0.5f, 1.f}, {0.5f, 0.5f}, {0.5f, 0.f}, {0.f, 0.5f}, {1.f, 0.5f},
        {0.f, 1.f},  {1.f, 1.f},   {0.f, 0.f},  {1.f, 0.f}};
    const float* f = kFraction[refIndex(ref)];
    return {box.x + f[0] * box.width, box.y + f[1] * box.height};
}

bool pointInPolygon(const cv::Point2f& p, const std::vector<cv::Point2f>& polygon) {
    if (polygon.size() < 3) return false;
    bool inside = false;
    for (size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
        const double xi = polygon[i].x, yi = polygon[i].y;
        const double xj = polygon[j].x, yj = polygon[j].y;
        if ((yi > p.y) != (yj > p.y) && p.x < (xj - xi) * (p.y - yi) / (yj - yi + 1e-12) + xi) {
            inside = !inside;
        }
    }
    return inside;
}

double median(const std::deque<float>& values) {
    if (values.empty()) return 0.0;
    std::vector<float> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end());
    const size_t n = sorted.size();
    return n % 2 ? sorted[n / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
}

double distance(const cv::Point2f& a, const cv::Point2f& b) { return std::hypot(a.x - b.x, a.y - b.y); }

Side opposite(Side side) { return side == Side::Inside ? Side::Outside : Side::Inside; }

Side toSide(Region region) { return region == Region::Inside ? Side::Inside : Side::Outside; }

Region toRegion(Side side) { return side == Side::Inside ? Region::Inside : Region::Outside; }

EventType eventToward(Side destination) {
    return destination == Side::Inside ? EventType::In : EventType::Out;
}

struct RegionKey {
    Region region;
    Side leads;
    bool far;
    bool operator==(const RegionKey& o) const {
        return region == o.region && leads == o.leads && far == o.far;
    }
};

struct LineState {
    int side = 0;
    cv::Point2f point;
};

struct TrackState {
    int64_t id = 0;
    int64_t lastFrame = kNoFrame;
    std::deque<cv::Point2f> trail;
    std::array<std::deque<cv::Point2f>, kRefCount> raws;
    std::array<cv::Point2f, kRefCount> points{};
    std::deque<float> heights;
    cv::Point2f center;
    float smoothW = 0.f;
    float smoothH = 0.f;
    std::optional<float> sizeRatio;
    std::map<std::string, LineState> lines;
    std::optional<RegionKey> regionRaw;
    int regionRun = 0;
    std::optional<Region> regionStable;
    Side stableLeads = Side::Outside;
    bool stableFar = false;
    std::optional<Region> lastRegion;
    Side lastLeads = Side::Outside;
    std::string portalId = "region";
    std::optional<Side> side;
    cv::Point2f sidePoint;
    bool vanishChecked = false;
    std::optional<EventType> lastEvent;
};

struct Classification {
    Region region = Region::Inside;
    Side leads = Side::Inside;
    std::string zoneId;
    bool far = false;
    std::optional<float> ratio;
};

void resetRegionState(TrackState& st) {
    st.sizeRatio.reset();
    st.regionRaw.reset();
    st.regionRun = 0;
    st.regionStable.reset();
    st.stableLeads = Side::Outside;
    st.stableFar = false;
    st.lastRegion.reset();
    st.lastLeads = Side::Outside;
    st.portalId = "region";
    st.side.reset();
    st.sidePoint = cv::Point2f();
    st.vanishChecked = false;
}

}

struct ZoneObjectCounter::Impl {
    CounterConfig cfg;
    std::map<int64_t, TrackState> tracks;
    std::map<std::string, GateCounts> counts;
    int totalIn = 0;
    int totalOut = 0;
    int64_t lastFrame = kNoFrame;

    ActiveRefs activeRefs() const {
        ActiveRefs active{};
        active[refIndex(cfg.refPoint)] = true;
        for (const auto& line : cfg.lines) {
            if (line.refPoint) active[refIndex(*line.refPoint)] = true;
        }
        for (const auto& zone : cfg.region.zones) {
            if (zone.refPoint) active[refIndex(*zone.refPoint)] = true;
        }
        return active;
    }

    static float thresholdFor(float height, float px, float ratio) { return std::max(px, ratio * height); }

    void dropLineState(const std::string& id) {
        for (auto& entry : tracks) entry.second.lines.erase(id);
    }

    void dropAllLineStates() {
        for (auto& entry : tracks) entry.second.lines.clear();
    }

    void resetAllRegionStates() {
        for (auto& entry : tracks) resetRegionState(entry.second);
    }

    Side defaultSide() const {
        const auto& rc = cfg.region;
        if (rc.defaultSide == DefaultSide::Inside) return Side::Inside;
        if (rc.defaultSide == DefaultSide::Outside) return Side::Outside;
        bool hasInside = false, hasOutside = false;
        for (const auto& zone : rc.zones) {
            hasInside = hasInside || zone.kind == ZoneKind::Inside;
            hasOutside = hasOutside || zone.kind == ZoneKind::Outside;
        }
        if (hasInside && !hasOutside) return Side::Outside;
        if (hasOutside && !hasInside) return Side::Inside;
        for (const auto& zone : rc.zones) {
            if (zone.kind == ZoneKind::Portal) return opposite(zone.leadsTo);
        }
        return Side::Inside;
    }

    void push(TrackState& st, int64_t frame, const cv::Rect2f& box, const ActiveRefs& active) {
        st.lastFrame = frame;
        const float w = std::max(1.f, box.width);
        const float h = std::max(1.f, box.height);
        if (st.smoothH == 0.f) {
            st.smoothW = w;
            st.smoothH = h;
        } else {
            st.smoothW = 0.6f * st.smoothW + 0.4f * w;
            st.smoothH = 0.6f * st.smoothH + 0.4f * h;
        }
        st.heights.push_back(h);
        while (st.heights.size() > kHeightWindow) st.heights.pop_front();
        st.center = {box.x + 0.5f * box.width, box.y + 0.5f * box.height};

        const size_t window = static_cast<size_t>(std::max(1, cfg.smoothWindow));
        for (int i = 0; i < kRefCount; ++i) {
            if (!active[i]) continue;
            auto& raw = st.raws[i];
            raw.push_back(referencePoint(box, static_cast<RefPoint>(i)));
            while (raw.size() > window) raw.pop_front();
            float sx = 0.f, sy = 0.f;
            for (const auto& p : raw) {
                sx += p.x;
                sy += p.y;
            }
            st.points[i] = {sx / raw.size(), sy / raw.size()};
        }
        st.trail.push_back(st.points[refIndex(cfg.refPoint)]);
        const size_t maxHistory = static_cast<size_t>(std::max(1, cfg.maxHistory));
        while (st.trail.size() > maxHistory) st.trail.pop_front();
    }

    void emit(std::vector<ZoneEvent>& out, TrackState& st, EventType type, const std::string& gateId,
              int64_t frame, const cv::Point2f& point) {
        GateCounts& gate = counts[gateId];
        if (type == EventType::In) {
            ++totalIn;
            ++gate.in;
        } else {
            ++totalOut;
            ++gate.out;
        }
        st.lastEvent = type;
        out.push_back({type, st.id, frame, point, gateId});
    }

    std::optional<EventType> lineStep(TrackState& st, const LineGateConfig& line, const cv::Point2f& cur,
                                      float height) {
        const double dx = line.p1.x - line.p0.x;
        const double dy = line.p1.y - line.p0.y;
        const double length = std::hypot(dx, dy);
        if (length < 1e-6) return std::nullopt;
        const double nx = dy / length;
        const double ny = -dx / length;
        const double d = (cur.x - line.p0.x) * nx + (cur.y - line.p0.y) * ny;
        if (std::abs(d) < std::max<double>(line.marginPx, line.marginBBoxRatio * height)) return std::nullopt;

        const int side = d > 0 ? 1 : -1;
        LineState& ls = st.lines[line.id];
        if (ls.side == 0 || side == ls.side) {
            ls.side = side;
            ls.point = cur;
            return std::nullopt;
        }
        const double d0 = (ls.point.x - line.p0.x) * nx + (ls.point.y - line.p0.y) * ny;
        const double denominator = d0 - d;
        const cv::Point2f previous = ls.point;
        ls.side = side;
        ls.point = cur;
        if (std::abs(denominator) < 1e-9) return std::nullopt;

        const double t = d0 / denominator;
        const double ix = previous.x + t * (cur.x - previous.x);
        const double iy = previous.y + t * (cur.y - previous.y);
        const double u = ((ix - line.p0.x) * dx + (iy - line.p0.y) * dy) / (length * length);
        if (u < -line.extend || u > 1.0 + line.extend) return std::nullopt;

        const int insideSign = line.inSide == LineSide::Left ? 1 : -1;
        return side == insideSign ? EventType::In : EventType::Out;
    }

    Classification classify(const TrackState& st) const {
        const auto& rc = cfg.region;
        for (const auto& zone : rc.zones) {
            if (zone.kind != ZoneKind::Portal) continue;
            const cv::Point2f& p = st.points[refIndex(zone.refPoint.value_or(cfg.refPoint))];
            const bool inside = pointInPolygon(p, zone.polygon) ||
                                (rc.portalUseBoxCenter && pointInPolygon(st.center, zone.polygon));
            if (!inside) continue;

            float minX = zone.polygon.front().x, maxX = minX;
            float minY = zone.polygon.front().y, maxY = minY;
            for (const auto& q : zone.polygon) {
                minX = std::min(minX, q.x);
                maxX = std::max(maxX, q.x);
                minY = std::min(minY, q.y);
                maxY = std::max(maxY, q.y);
            }
            const double rectW = maxX - minX;
            const double rectH = maxY - minY;
            Classification c;
            c.zoneId = zone.id;
            c.leads = zone.leadsTo;
            if (rectW > 1e-6 && rectH > 1e-6) {
                c.ratio = zone.farMetric == FarMetric::Height
                              ? static_cast<float>(st.smoothH / rectH)
                              : static_cast<float>((static_cast<double>(st.smoothW) * st.smoothH) / (rectW * rectH));
            }
            if (zone.farPercent > 0.f && c.ratio && *c.ratio < zone.farPercent / 100.f) {
                c.region = toRegion(zone.leadsTo);
                c.far = true;
            } else {
                c.region = Region::Portal;
            }
            return c;
        }
        for (const auto& zone : rc.zones) {
            if (zone.kind == ZoneKind::Portal) continue;
            const cv::Point2f& p = st.points[refIndex(zone.refPoint.value_or(cfg.refPoint))];
            if (!pointInPolygon(p, zone.polygon)) continue;
            Classification c;
            c.region = zone.kind == ZoneKind::Inside ? Region::Inside : Region::Outside;
            c.leads = toSide(c.region);
            c.zoneId = zone.id;
            return c;
        }
        Classification c;
        const Side fallback = defaultSide();
        c.region = toRegion(fallback);
        c.leads = fallback;
        c.zoneId = "region";
        return c;
    }

    std::optional<EventType> regionStep(TrackState& st, const cv::Point2f& cur, float height) {
        const auto& rc = cfg.region;
        const Classification c = classify(st);
        st.sizeRatio = c.ratio;
        if (c.region == Region::Portal || c.far) st.portalId = c.zoneId;
        if (c.region == Region::Portal && !st.regionRaw) {
            st.side = c.leads;
            st.sidePoint = cur;
        }
        st.lastRegion = c.region;
        st.lastLeads = c.leads;

        const RegionKey key{c.region, c.leads, c.far};
        if (st.regionRaw && *st.regionRaw == key) {
            ++st.regionRun;
        } else {
            st.regionRaw = key;
            st.regionRun = 1;
        }
        if (st.regionRun >= rc.minDwellFrames) {
            st.regionStable = c.region;
            st.stableLeads = c.leads;
            st.stableFar = c.far;
        }
        if (!st.regionStable) return std::nullopt;

        const Region stable = *st.regionStable;
        if (stable == Region::Portal) {
            if (!st.side || *st.side == st.stableLeads) {
                st.side = st.stableLeads;
                st.sidePoint = cur;
            }
            return std::nullopt;
        }
        const Side stableSide = toSide(stable);
        if (!st.side || *st.side == stableSide) {
            st.side = stableSide;
            st.sidePoint = cur;
            return std::nullopt;
        }
        if (st.stableFar ||
            distance(cur, st.sidePoint) >= thresholdFor(height, rc.minTravelPx, rc.minTravelBBoxRatio)) {
            st.side = stableSide;
            st.sidePoint = cur;
            return eventToward(stableSide);
        }
        return std::nullopt;
    }

    std::optional<EventType> vanish(TrackState& st) {
        if (!st.lastRegion || *st.lastRegion != Region::Portal || !st.side || *st.side == st.lastLeads) {
            return std::nullopt;
        }
        st.side = st.lastLeads;
        st.sidePoint = st.trail.back();
        return eventToward(st.lastLeads);
    }

    std::vector<ZoneEvent> process(int64_t frame, const std::vector<TrackObservation>& observations) {
        std::vector<ZoneEvent> events;
        std::set<int64_t> seen;
        const ActiveRefs active = activeRefs();
        const bool hasZones = !cfg.region.zones.empty();

        for (const auto& obs : observations) {
            if (!seen.insert(obs.trackId).second) continue;
            auto it = tracks.find(obs.trackId);
            if (it != tracks.end() && it->second.lastFrame >= frame) continue;
            if (it == tracks.end()) {
                it = tracks.emplace(obs.trackId, TrackState{}).first;
                it->second.id = obs.trackId;
            }
            TrackState& st = it->second;
            push(st, frame, obs.box, active);
            st.vanishChecked = false;

            const cv::Point2f cur = st.trail.back();
            const float height = static_cast<float>(median(st.heights));
            for (const auto& line : cfg.lines) {
                const cv::Point2f p = st.points[refIndex(line.refPoint.value_or(cfg.refPoint))];
                if (auto type = lineStep(st, line, p, height)) emit(events, st, *type, line.id, frame, p);
            }
            if (hasZones) {
                if (auto type = regionStep(st, cur, height)) emit(events, st, *type, st.portalId, frame, cur);
            }
        }

        for (auto it = tracks.begin(); it != tracks.end();) {
            TrackState& st = it->second;
            if (seen.count(it->first)) {
                ++it;
                continue;
            }
            const int64_t missed = frame - st.lastFrame;
            if (hasZones && !st.vanishChecked &&
                missed >= std::min(cfg.region.commitLostFrames, cfg.trackTimeoutFrames)) {
                st.vanishChecked = true;
                if (auto type = vanish(st)) emit(events, st, *type, st.portalId, frame, st.trail.back());
            }
            if (missed > cfg.trackTimeoutFrames) {
                it = tracks.erase(it);
            } else {
                ++it;
            }
        }
        lastFrame = std::max(lastFrame, frame);
        return events;
    }

    std::vector<ZoneEvent> processHistories(const std::vector<TrackHistory>& histories,
                                            std::optional<int64_t> currentFrame) {
        std::map<int64_t, std::vector<TrackObservation>> byFrame;
        for (const auto& history : histories) {
            for (const auto& obs : history.observations) {
                if (obs.frame <= lastFrame) continue;
                TrackObservation item = obs;
                item.trackId = history.trackId;
                byFrame[obs.frame].push_back(item);
            }
        }
        std::vector<ZoneEvent> events;
        for (const auto& entry : byFrame) {
            auto frameEvents = process(entry.first, entry.second);
            events.insert(events.end(), frameEvents.begin(), frameEvents.end());
        }
        if (currentFrame && *currentFrame > lastFrame) {
            auto frameEvents = process(*currentFrame, {});
            events.insert(events.end(), frameEvents.begin(), frameEvents.end());
        }
        return events;
    }
};

ZoneObjectCounter::ZoneObjectCounter(CounterConfig config) : impl_(std::make_unique<Impl>()) {
    impl_->cfg = std::move(config);
}

ZoneObjectCounter::~ZoneObjectCounter() = default;

ZoneObjectCounter::ZoneObjectCounter(ZoneObjectCounter&&) noexcept = default;

ZoneObjectCounter& ZoneObjectCounter::operator=(ZoneObjectCounter&&) noexcept = default;

void ZoneObjectCounter::setConfig(CounterConfig config) {
    impl_->cfg = std::move(config);
    impl_->dropAllLineStates();
    impl_->resetAllRegionStates();
}

const CounterConfig& ZoneObjectCounter::config() const { return impl_->cfg; }

bool ZoneObjectCounter::addLine(LineGateConfig line) {
    if (line.id.empty()) return false;
    for (const auto& existing : impl_->cfg.lines) {
        if (existing.id == line.id) return false;
    }
    impl_->dropLineState(line.id);
    impl_->cfg.lines.push_back(std::move(line));
    return true;
}

bool ZoneObjectCounter::removeLine(const std::string& id) {
    auto& lines = impl_->cfg.lines;
    const auto it = std::find_if(lines.begin(), lines.end(), [&](const LineGateConfig& l) { return l.id == id; });
    if (it == lines.end()) return false;
    lines.erase(it);
    impl_->dropLineState(id);
    return true;
}

bool ZoneObjectCounter::addZone(ZoneConfig zone) {
    if (zone.id.empty() || zone.polygon.size() < 3) return false;
    for (const auto& existing : impl_->cfg.region.zones) {
        if (existing.id == zone.id) return false;
    }
    impl_->cfg.region.zones.push_back(std::move(zone));
    impl_->resetAllRegionStates();
    return true;
}

bool ZoneObjectCounter::removeZone(const std::string& id) {
    auto& zones = impl_->cfg.region.zones;
    const auto it = std::find_if(zones.begin(), zones.end(), [&](const ZoneConfig& z) { return z.id == id; });
    if (it == zones.end()) return false;
    zones.erase(it);
    impl_->resetAllRegionStates();
    return true;
}

void ZoneObjectCounter::clearGates() {
    impl_->cfg.lines.clear();
    impl_->cfg.region.zones.clear();
    impl_->dropAllLineStates();
    impl_->resetAllRegionStates();
}

std::vector<ZoneEvent> ZoneObjectCounter::update(int64_t frame, const std::vector<TrackObservation>& observations) {
    return impl_->process(frame, observations);
}

std::vector<ZoneEvent> ZoneObjectCounter::updateFromHistories(const std::vector<TrackHistory>& histories,
                                                              std::optional<int64_t> currentFrame) {
    return impl_->processHistories(histories, currentFrame);
}

void ZoneObjectCounter::reset() {
    impl_->tracks.clear();
    resetCounts();
    impl_->lastFrame = kNoFrame;
}

void ZoneObjectCounter::resetCounts() {
    impl_->counts.clear();
    impl_->totalIn = 0;
    impl_->totalOut = 0;
}

int ZoneObjectCounter::countIn() const { return impl_->totalIn; }

int ZoneObjectCounter::countOut() const { return impl_->totalOut; }

const std::map<std::string, GateCounts>& ZoneObjectCounter::gateCounts() const { return impl_->counts; }

Side ZoneObjectCounter::defaultSide() const { return impl_->defaultSide(); }

std::vector<int64_t> ZoneObjectCounter::trackIds() const {
    std::vector<int64_t> ids;
    ids.reserve(impl_->tracks.size());
    for (const auto& entry : impl_->tracks) ids.push_back(entry.first);
    return ids;
}

std::optional<TrackInfo> ZoneObjectCounter::trackInfo(int64_t trackId) const {
    const auto it = impl_->tracks.find(trackId);
    if (it == impl_->tracks.end()) return std::nullopt;
    const TrackState& st = it->second;
    TrackInfo info;
    info.trackId = st.id;
    info.lastFrame = st.lastFrame;
    info.trail.assign(st.trail.begin(), st.trail.end());
    info.side = st.side;
    info.region = st.lastRegion;
    info.sizeRatio = st.sizeRatio;
    info.lastEvent = st.lastEvent;
    return info;
}

}
