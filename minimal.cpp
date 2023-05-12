// Define macro to enable function definitions for the header only library
// To learn more, see https://github.com/nothings/stb#how-do-i-use-these-libraries
#define CPPKALMAN_IMPL
#include "cppkalman.hpp"

#include <iostream>

int main()
{
    // A one-dimensional constant velocity model
    // State vector:       [xp, xv]
    // Observation vector: [xp]
    // The template parameters <2, 1> are the sizes of these vectors, respectively
    using KF = cppkalman::AdditiveUnscentedKalmanFilter<2, 1>;

    // Define (potentially non-linear) transition and observation functions
    auto transitionFunc = [](KF::State state) -> KF::State {
        state(0) += state(1); // xp += xv
        return state;
    };
    auto observationFunc = [](KF::State state) -> KF::Observation {
        return state.head(1); // xp
    };
    KF kf{transitionFunc, observationFunc};

    // Predict
    cppkalman::Moments<2> currentState{
        KF::State{ 2.2, 1.1 }, // Predicted position, velocity
        Eigen::Matrix2d{{ 1.0, 0.0 },
                        { 0.0, 1.0 }}, // State covariance
    };
    auto [predictedState, sigmaPoints] = kf.Predict(currentState);

    // Update
    KF::Observation currentObservation{ 3.5 }; // Observed position
    Eigen::Matrix<double, 1, 1> R{ 1.0 }; // Observation covariance
    auto nextState = kf.Update(predictedState, sigmaPoints, R, currentObservation);

    std::cout << observationFunc(nextState.stateMean) << '\n'; // Filtered position
    // The filtered position should be between predicted (3.3) and obsereved (3.5)
}
