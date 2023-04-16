#define CPPKALMAN_IMPL
#include "cppkalman.hpp"
using namespace cppkalman;

#include <iostream>

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
    0.0, 0.0, 1.0, 0.0, 0.0, 0,0
).finished();


KalmanFilter::State Transition(KalmanFilter::State state)
{
    return F * state;
}

KalmanFilter::Observation Observation(KalmanFilter::State state)
{
    return H * state;
}


int main()
{
    KalmanFilter kf{Transition, Observation};

    KalmanFilter::Observation obs{{ 8.892125, -3.183644, -0.766539 }};
    Moments<STATE_SIZE, T> current;
    current.stateMean << 8.892125, -3.183644, -0.766539, 0., 0., 0.;
    current.stateCovariance << 
        100,   0,     0,     0,     0,     0,
        0, 10000,     0,     0,     0,     0,
        0,     0,   100,     0,     0,     0,
        0,     0,     0, 10000,     0,     0,
        0,     0,     0,     0,   100,     0,
        0,     0,     0,     0,     0, 10000;

    auto [moments, points] = kf.Predict(current);
    Moments<STATE_SIZE, T> next = kf.Update(moments, points, obs);
    (void) next;
    std::cout << "OK!\n";
}
