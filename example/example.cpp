#include "example_templates.hpp"
using namespace cppkalman;
using KalmanFilter = AdditiveUnscentedKalmanFilter<STATE_SIZE, OBSERVATION_SIZE, FloatType>;

#include <iostream>
#include <fstream>


static constexpr FloatType dt = 1.0;
// Unfortunately Eigen does not have constexpr constructors yet
static const Eigen::Matrix<FloatType, STATE_SIZE, STATE_SIZE> F = (Eigen::Matrix<FloatType, STATE_SIZE, STATE_SIZE>() <<
    1.0, 0.0, 0.0,  dt, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,  dt, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0,  dt,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0
).finished();

static const Eigen::Matrix<FloatType, OBSERVATION_SIZE, STATE_SIZE> H = (Eigen::Matrix<FloatType, OBSERVATION_SIZE, STATE_SIZE>() <<
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0
).finished();

static KalmanFilter::State TransitionFunction(KalmanFilter::State state)
{
    return F * state;
}

static KalmanFilter::Observation ObservationFunction(KalmanFilter::State state)
{
    return H * state;
}


#define DISTANCE_THRESHOLD 2.0
#define DELETE_AGE_THRESHOLD 3

struct Track
{
    static size_t _id;
    Track(size_t, const KalmanFilter::Observation&);

    size_t id;
    size_t startTime = 0;
    size_t endTime = 0;
    Moments<STATE_SIZE, FloatType> current;
    size_t age = 0;
    bool isCoasting = false;
    // bool isConfirmed = false;
    // std::vector<KalmanFilter::State> history;
    KalmanFilter kf;
};
size_t Track::_id = 0;

Track::Track(size_t time, const KalmanFilter::Observation& obs)
    : id{_id++}
    , startTime{time}
    , kf{TransitionFunction, ObservationFunction}
{
    current.stateMean << obs(0), obs(1), obs(2), 0, 0, 0;
    current.stateCovariance << 
      100,     0,     0,     0,     0,     0,
        0,   100,     0,     0,     0,     0,
        0,     0,   100,     0,     0,     0,
        0,     0,     0, 10000,     0,     0,
        0,     0,     0,     0, 10000,     0,
        0,     0,     0,     0,     0, 10000;

    auto [momentsPred, pointsPred] = kf.Predict(current);
    current = kf.Update(momentsPred, pointsPred, obs);
    // history.push_back(current.stateMean);
}

static FloatType Dist(FloatType x1, FloatType y1, FloatType z1, FloatType x2, FloatType y2, FloatType z2)
{
    return std::sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1));
}

static void StepGNN(
    size_t t,
    std::vector<KalmanFilter::Observation> observations,
    std::vector<Track>& tracks,
    std::vector<Track>& deadTracks)
{

    if (t == 0) {
        for (const auto& observation : observations)
            tracks.emplace_back(t, observation);
        return;
    }

    static std::vector<size_t> toDelete;
    toDelete.clear();
    for (size_t trackIdx = 0; trackIdx < tracks.size(); ++trackIdx) {
        auto& track = tracks[trackIdx];

        // ==== PREDICT ====
        auto [momentsPred, pointsPred] = track.kf.Predict(track.current);

        // ==== ASSOCIATE ====
        FloatType bestDst = 1e9;
        size_t bestIdx = observations.size();
        for (size_t obsIdx = 0; obsIdx < observations.size(); ++obsIdx) {
            const auto& observation = observations[obsIdx];
            FloatType dst = Dist(momentsPred.stateMean(0), momentsPred.stateMean(1), momentsPred.stateMean(2),
                                 observation(0), observation(1), observation(2));
            if (dst < DISTANCE_THRESHOLD && dst < bestDst) {
                bestDst = dst;
                bestIdx = obsIdx;
            }
        }

        std::optional<KalmanFilter::Observation> observation = std::nullopt;
        if (bestIdx != observations.size()) {
            observation = observations[bestIdx];
            observations.erase(observations.begin() + bestIdx);
            track.age = 0;
            track.isCoasting = false;
        }
        else {
            track.age += 1;
            track.isCoasting = true;
        }

        if (track.age >= DELETE_AGE_THRESHOLD) {
            toDelete.push_back(trackIdx);
        }
        track.endTime = t;
        if (t > track.startTime + 5) { // FIXME: Use rolling average of detection rate
            track.isConfirmed = true;
        }

        // ==== UPDATE ====
        track.current = track.kf.Update(momentsPred, pointsPred, observation);

        // Save result for plotting
        // track.history.push_back(track.current.stateMean);
    }

    // Delete old tracks
    for (size_t i = toDelete.size()-1; i+1 > 0; --i) {
        size_t trackIdx = toDelete[i];
        deadTracks.emplace_back(std::move(tracks[trackIdx]));
        tracks.erase(tracks.begin() + (long)trackIdx);
    }

    // Any remaining unmatched observations
    for (const auto& observation : observations)
        tracks.emplace_back(t, observation);
}

static std::vector<std::vector<KalmanFilter::Observation>> GetTestData()
{
    std::vector<std::vector<KalmanFilter::Observation>> all_observations;
    all_observations.emplace_back();
    std::ifstream inp{"sample.txt"};
    
    std::string s;
    while (std::getline(inp, s)) {
        if (s.empty() || std::isspace(s[0])) {
            all_observations.emplace_back();
            continue;
        }
        auto& obs = all_observations.back().emplace_back();
        sscanf(s.c_str(), "%f %f %f", &obs(0), &obs(1), &obs(2));
    }

    return all_observations;
}

int main()
{
    const auto all_observations = GetTestData();
    std::vector<Track> tracks;
    std::vector<Track> deadTracks;

    for (size_t t = 0; t < all_observations.size(); ++t) {
        StepGNN(t, all_observations[t], tracks, deadTracks);
    }
}
