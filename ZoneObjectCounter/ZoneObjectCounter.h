#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace zoc {

enum class EventType { In, Out };

enum class Side { Inside, Outside };

enum class Region { Inside, Outside, Portal };

enum class ZoneKind { Portal, Inside, Outside };

enum class LineSide { Left, Right };

enum class FarMetric { Area, Height };

enum class DefaultSide { Auto, Inside, Outside };

enum class RefPoint {
    BottomCenter,
    Center,
    TopCenter,
    LeftCenter,
    RightCenter,
    BottomLeft,
    BottomRight,
    TopLeft,
    TopRight
};

struct LineGateConfig {
    std::string id;
    cv::Point2f p0;
    cv::Point2f p1;
    LineSide inSide = LineSide::Left;
    float marginPx = 8.f;
    float marginBBoxRatio = 0.08f;
    float extend = 0.15f;
    std::optional<RefPoint> refPoint;
};

struct ZoneConfig {
    std::string id;
    ZoneKind kind = ZoneKind::Portal;
    std::vector<cv::Point2f> polygon;
    Side leadsTo = Side::Outside;
    float farPercent = 0.f;
    FarMetric farMetric = FarMetric::Area;
    std::optional<RefPoint> refPoint;
};

struct RegionConfig {
    std::vector<ZoneConfig> zones;
    DefaultSide defaultSide = DefaultSide::Auto;
    int minDwellFrames = 3;
    int commitLostFrames = 8;
    bool portalUseBoxCenter = true;
    float minTravelPx = 30.f;
    float minTravelBBoxRatio = 0.35f;
};

struct CounterConfig {
    std::vector<LineGateConfig> lines;
    RegionConfig region;
    RefPoint refPoint = RefPoint::BottomCenter;
    int smoothWindow = 5;
    int trackTimeoutFrames = 30;
    int maxHistory = 300;
};

struct TrackObservation {
    int64_t trackId = 0;
    int64_t frame = 0;
    cv::Rect2f box;
};

struct TrackHistory {
    int64_t trackId = 0;
    std::vector<TrackObservation> observations;
};

struct ZoneEvent {
    EventType type = EventType::In;
    int64_t trackId = 0;
    int64_t frame = 0;
    cv::Point2f point;
    std::string gateId;
};

struct GateCounts {
    int in = 0;
    int out = 0;
};

struct TrackInfo {
    int64_t trackId = 0;
    int64_t lastFrame = 0;
    std::vector<cv::Point2f> trail;
    std::optional<Side> side;
    std::optional<Region> region;
    std::optional<float> sizeRatio;
    std::optional<EventType> lastEvent;
};

class ZoneObjectCounter {
public:
    explicit ZoneObjectCounter(CounterConfig config = CounterConfig{});
    ~ZoneObjectCounter();

    ZoneObjectCounter(const ZoneObjectCounter&) = delete;
    ZoneObjectCounter& operator=(const ZoneObjectCounter&) = delete;
    ZoneObjectCounter(ZoneObjectCounter&&) noexcept;
    ZoneObjectCounter& operator=(ZoneObjectCounter&&) noexcept;

    void setConfig(CounterConfig config);
    const CounterConfig& config() const;

    bool addLine(LineGateConfig line);
    bool removeLine(const std::string& id);
    bool addZone(ZoneConfig zone);
    bool removeZone(const std::string& id);
    void clearGates();

    std::vector<ZoneEvent> update(int64_t frame, const std::vector<TrackObservation>& observations);
    std::vector<ZoneEvent> updateFromHistories(const std::vector<TrackHistory>& histories,
                                               std::optional<int64_t> currentFrame = std::nullopt);

    void reset();
    void resetCounts();

    int countIn() const;
    int countOut() const;
    const std::map<std::string, GateCounts>& gateCounts() const;

    Side defaultSide() const;
    std::vector<int64_t> trackIds() const;
    std::optional<TrackInfo> trackInfo(int64_t trackId) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
