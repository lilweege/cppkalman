#define CPPKALMAN_IMPL
#include "cppkalman.hpp"
using namespace cppkalman;

#include <iostream>
#include <fstream>

constexpr size_t STATE_SIZE = 6;
constexpr size_t OBSERVATION_SIZE = 3;
using T = double;
using KalmanFilter = AdditiveUnscentedKalmanFilter<STATE_SIZE, OBSERVATION_SIZE, T>;


constexpr T dt = 1.0;
// Unfortunately Eigen does not have constexpr constructors yet
static const Eigen::Matrix<T, STATE_SIZE, STATE_SIZE> F = (Eigen::Matrix<T, STATE_SIZE, STATE_SIZE>() <<
    1.0, 0.0, 0.0,  dt, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,  dt, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0,  dt,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0
).finished();

static const Eigen::Matrix<T, OBSERVATION_SIZE, STATE_SIZE> H = (Eigen::Matrix<T, OBSERVATION_SIZE, STATE_SIZE>() <<
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0
).finished();


KalmanFilter::State Transition(KalmanFilter::State state)
{
    return F * state;
}

KalmanFilter::Observation Observation(KalmanFilter::State state)
{
    return H * state;
}


#define DISTANCE_THRESHOLD 1.0
#define DELETE_AGE_THRESHOLD 3

struct Track
{
    static size_t _id;
    Track(size_t, const KalmanFilter::Observation&);

    size_t id;
    size_t startTime = 0;
    size_t endTime = 0;
    Moments<STATE_SIZE, T> current;
    size_t age = 0;
    bool isCoasting = false;
    bool isConfirmed = false;
    std::vector<KalmanFilter::State> history;
    KalmanFilter kf;
};
size_t Track::_id = 0;

Track::Track(size_t time, const KalmanFilter::Observation& obs)
    : id{_id++}
    , startTime{time}
    , kf{Transition, Observation}
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
    history.push_back(current.stateMean);
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
        sscanf(s.c_str(), "%lf %lf %lf", &obs(0), &obs(1), &obs(2));
    }

    return all_observations;
}

static T Dist(T x1, T y1, T z1, T x2, T y2, T z2)
{
    return std::sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1));
}


static std::vector<Track> tracks, deadTracks;
static void Step(size_t t, std::vector<KalmanFilter::Observation> observations)
{
    if (t == 0) {
        for (const auto& observation : observations)
            tracks.emplace_back(t, observation);
        return;
    }

    std::vector<size_t> toDelete;
    for (size_t trackIdx = 0; trackIdx < tracks.size(); ++trackIdx) {
        auto& track = tracks[trackIdx];

        // ==== PREDICT ====
        auto [momentsPred, pointsPred] = track.kf.Predict(track.current);

        // ==== ASSOCIATE ====
        T bestDst = 1e9;
        size_t bestIdx = observations.size();
        for (size_t obsIdx = 0; obsIdx < observations.size(); ++obsIdx) {
            const auto& observation = observations[obsIdx];
            T dst = Dist(momentsPred.stateMean(0), momentsPred.stateMean(1), momentsPred.stateMean(2),
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
        }
        else {
            track.age += 1;
        }

        if (track.age >= DELETE_AGE_THRESHOLD) {
            toDelete.push_back(trackIdx);
        }
        track.endTime = t;

        // ==== UPDATE ====
        track.current = track.kf.Update(momentsPred, pointsPred, observation);

        // Save result for plotting
        track.history.push_back(track.current.stateMean);
    }

    // Delete old tracks
    for (size_t i = toDelete.size()-1; i+1 > 0; --i) {
        size_t trackIdx = toDelete[i];
        deadTracks.emplace_back(std::move(tracks[trackIdx]));
        tracks.erase(tracks.begin() + trackIdx);
    }

    // Any remaining unmatched observations
    for (const auto& observation : observations)
        tracks.emplace_back(t, observation);
}


int main()
{
    const std::vector<std::vector<KalmanFilter::Observation>> all_observations = GetTestData();

    for (size_t t = 0; t < all_observations.size(); ++t) {
        Step(t, all_observations[t]);
        // ...
    }
}
