#define CPPKALMAN_IMPL
#include "cppkalman.hpp"
using namespace cppkalman;

#include <iostream>

constexpr size_t STATE_SIZE = 6;
constexpr size_t OBSERVATION_SIZE = 3;
using KalmanFilter = AdditiveUnscentedKalmanFilter<STATE_SIZE, OBSERVATION_SIZE>;


// Unfortunately Eigen does not have constexpr constructors yet
KalmanFilter::State Transition(KalmanFilter::State state)
{
    constexpr double dt = 1.0;
    static const Eigen::Matrix<double, STATE_SIZE, STATE_SIZE> F{{
        1.0, 0.0, 0.0,  dt, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,  dt, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0,  dt,
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    }};
    return F * state;
}

KalmanFilter::Observation Observation(KalmanFilter::State state)
{
    static const Eigen::Matrix<double, OBSERVATION_SIZE, STATE_SIZE> H{{
        1.0, 0.0, 0.0, 0,0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0,0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 0,0,
    }};
    return H * state;
}


int main()
{
    KalmanFilter kf{Transition, Observation};

    KalmanFilter::Sample s;
    kf.Predict(s);
    s = kf.Update(std::nullopt);
}
