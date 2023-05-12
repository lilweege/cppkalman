/* cppkalman - v1.0 - A port of pykalman's AdditiveUnscentedKalmanFilter to C++
This library is header-only (see https://github.com/nothings/stb#how-do-i-use-these-libraries)

Luigi Quattrociocchi - April 20, 2023
*/


#ifndef CPPKALMAN_H
#define CPPKALMAN_H

#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include <functional>
#include <utility>

namespace cppkalman {

template<int N, typename T=double>
struct Moments
{
    Eigen::Vector<T, N> stateMean;
    Eigen::Matrix<T, N, N> stateCovariance;
};

template<int StateSize, int N, typename T=double>
struct SigmaPoints
{
    Eigen::Matrix<T, 2*StateSize+1, N> points;
    Eigen::Vector<T, 2*StateSize+1> weightsMean;
    Eigen::Vector<T, 2*StateSize+1> weightsCovariance;
};

/**
 * A port of pykalman's AdditiveUnscentedKalmanFilter.
 *
 * @tparam StateSize The number of elements in the state vector.
 * @tparam ObservationSize The number of elements in the observation vector.
 * @tparam T The desired floating point type, defaults to double.
 */
template<int StateSize, int ObservationSize, typename T=double>
class AdditiveUnscentedKalmanFilter
{
public:
    using StateCovariance = Eigen::Matrix<T, StateSize, StateSize>;
    using State = Eigen::Vector<T, StateSize>;
    using Observation = Eigen::Vector<T, ObservationSize>;

    using TransitionFunction = std::function<State(State)>;
    using ObservationFunction = std::function<Observation(State)>;

    /**
     * @param[in] transitionFunction A function describing how the state changes between times t and t+1.
     * @param[in] observationFunction A function characterizing how the observation at time t+1 is generated.
     */
    AdditiveUnscentedKalmanFilter(
        TransitionFunction&& transitionFunction,
        ObservationFunction&& observationFunction);

    /**
     * Predict next state distribution
     *
     * Using the sigma points representing the state at time t given observations
     * from time steps 0...t, calculate the predicted mean, covariance, and sigma
     * points for the state at time t+1.
     * 
     * @param[in] currentState The state at time step t given observations from time steps 0...t.
     * @param[in] transitionCovariance Transition noise covariance matrix. Also known as Q.
     * @returns A pair containings: (1) predicted mean state and covariance corresponding to time step t+1,
     * and (2) sigma points corresponding to the predicted state.
     */
    [[nodiscard]] std::pair<Moments<StateSize, T>, SigmaPoints<StateSize, StateSize, T>> Predict(
        const Moments<StateSize, T>& currentState,
        const Eigen::Matrix<T, StateSize, StateSize>& transitionCovariance);

    /**
     * Integrate new observation to correct state estimates
     * 
     * @param[in] predictedState Predicted mean state and covariance corresponding to time step t+1.
     * @param[in] predictedPoints Sigma points corresponding to predictedState.
     * @param[in] observationCovariance Observation noise covariance matrix. Also known as R.
     * @param[in] observation An observation at time t+1. If no observation is provided, predictedState is returned.
     * @returns Filtered mean state and covariance corresponding to time step t+1.
    */
    [[nodiscard]] Moments<StateSize, T> Update(
        const Moments<StateSize, T>& predictedState,
        const SigmaPoints<StateSize, StateSize, T>& predictedPoints,
        const Eigen::Matrix<T, ObservationSize, ObservationSize>& observationCovariance,
        const Observation& observation);

public:
    TransitionFunction mTransitionFunction;
    ObservationFunction mObservationFunction;
};


} // namespace cppkalman
#endif // CPPKALMAN_H



#ifdef CPPKALMAN_IMPL
namespace cppkalman {

/**
 * Calculate "sigma points" used in Unscented Kalman Filter.
 * 
 * @param[in] moments Mean and covariance of a multivariate normal.
 * @returns Sigma points and associated weights.
*/
template<int StateSize, typename T>
SigmaPoints<StateSize, StateSize, T>
static
Moments2Points(const Moments<StateSize, T>& moments)
{
    SigmaPoints<StateSize, StateSize, T> points;
    const auto& [mu, sigma] = moments;
    const T alpha = 1.0;
    const T beta = 0.0;
    const T kappa = 3.0 - StateSize;

    Eigen::LLT<Eigen::Matrix<T, StateSize, StateSize>, Eigen::Upper> llt{sigma};
    Eigen::Matrix<T, StateSize, StateSize> sigma2 = llt.matrixLLT().transpose();
    // Clear upper triangle
    for (int i = 0; i < StateSize; ++i)
        for (int j = i+1; j < StateSize; ++j)
            sigma2(i, j) = 0;

    const T lambda = (alpha * alpha) * (StateSize + kappa) - StateSize;
    const T c = StateSize + lambda;

    for (int i = 0; i < StateSize; ++i) {
        points.points(0, i) = mu(i);
        for (int j = 1; j < StateSize + 1; ++j)
            points.points(j, i) = mu(i) + sigma2(i, j-1) * (T) sqrt(c);
        for (int j = StateSize + 1; j < 2*StateSize+1; ++j)
            points.points(j, i) = mu(i) - sigma2(i, j-(StateSize+1)) * (T) sqrt(c);
    }

    points.weightsMean.setOnes();
    points.weightsMean(0) = lambda / c;
    for (int i = 1; i < 2*StateSize+1; ++i)
        points.weightsMean(i) = (T) 0.5 / c;

    points.weightsCovariance = points.weightsMean;
    points.weightsCovariance(0) = lambda / c + (1 - alpha * alpha + beta);

    return points;
}


/**
 * Calculate estimated mean and covariance of sigma points.
 * 
 * @param[in] sigmaPoints SigmaPoints object containing points and weights.
 * @param[in] sigmaNoise Additive noise covariance matrix, if any.
 * @returns Mean and covariance estimated using points.
*/
template<int StateSize, int ObservationSize, typename T>
Moments<ObservationSize, T>
static
Points2Moments(
    const SigmaPoints<StateSize, ObservationSize, T>& sigmaPoints,
    const Eigen::Matrix<T, ObservationSize, ObservationSize>& sigmaNoise)
{
    Moments<ObservationSize, T> moments;
    const auto& [points, weightsMean, weightsCovariance] = sigmaPoints;
    
    moments.stateMean = points.transpose() * weightsMean;
    Eigen::Matrix<T, ObservationSize, 2*StateSize+1> pointsDiff = points.transpose();
    for (int i = 0; i < ObservationSize; ++i)
        for (int j = 0; j < 2*StateSize+1; ++j)
            pointsDiff(i, j) -= moments.stateMean(i);


    moments.stateCovariance = pointsDiff * weightsCovariance.asDiagonal() * pointsDiff.transpose() + sigmaNoise;

    return moments;
}

/**
 * Apply the Unscented Transform to a set of points.
 * 
 * Apply f to points (with secondary argument points_noise, if available),
 * then approximate the resulting mean and covariance. If sigma_noise is
 * available, treat it as additional variance due to additive noise.
 * 
 * @param[in] sigmaPoints Points to pass into f's first argument and associated weights.
 * @param[in] sigmaNoise Covariance matrix for additive noise, if any.
 * @param[in] f transition function from time t to time t+1.
*/
template<int StateSize, int ObservationSize, typename T, typename Function>
std::pair<Moments<ObservationSize, T>, SigmaPoints<StateSize, ObservationSize, T>>
static
UnscentedTransform(
    const SigmaPoints<StateSize, StateSize, T>& sigmaPoints,
    const Eigen::Matrix<T, ObservationSize, ObservationSize>& sigmaNoise,
    Function f) // FIXME: constrain function with concept
{
    SigmaPoints<StateSize, ObservationSize, T> points;
    points.weightsCovariance = sigmaPoints.weightsCovariance;
    points.weightsMean = sigmaPoints.weightsMean;
    for (int i = 0; i < 2*StateSize+1; ++i) {
        Eigen::Vector<T, ObservationSize> row = f(sigmaPoints.points.row(i));
        for (int j = 0; j < ObservationSize; ++j)
            points.points(i, j) = row(j);
    }
    return std::make_pair(Points2Moments(points, sigmaNoise), points);
}


template<int StateSize, int ObservationSize, typename T>
AdditiveUnscentedKalmanFilter<StateSize, ObservationSize, T>::AdditiveUnscentedKalmanFilter(
    TransitionFunction&& transitionFunction,
    ObservationFunction&& observationFunction
)
    : mTransitionFunction{transitionFunction}
    , mObservationFunction{observationFunction}
{
}


template<int StateSize, int ObservationSize, typename T>
std::pair<Moments<StateSize, T>, SigmaPoints<StateSize, StateSize, T>>
AdditiveUnscentedKalmanFilter<StateSize, ObservationSize, T>::Predict(
    const Moments<StateSize, T>& currentState,
    const Eigen::Matrix<T, StateSize, StateSize>& transitionCovariance)
{
    SigmaPoints pointsState = Moments2Points(currentState);
    auto [moments, _] = UnscentedTransform<StateSize, StateSize, T>(pointsState, transitionCovariance, mTransitionFunction);
    return std::make_pair(moments, Moments2Points(moments));
}

template<int StateSize, int ObservationSize, typename T>
Moments<StateSize, T>
AdditiveUnscentedKalmanFilter<StateSize, ObservationSize, T>::Update(
    const Moments<StateSize, T>& predictedState,
    const SigmaPoints<StateSize, StateSize, T>& predictedPoints,
    const Eigen::Matrix<T, ObservationSize, ObservationSize>& observationCovariance,
    const Observation& observation)
{
    auto [obsMomentsPredicted, obsPointsPredicted] = UnscentedTransform<StateSize, ObservationSize, T>(predictedPoints, observationCovariance, mObservationFunction);

    auto predDiff = predictedPoints.points;
    for (int i = 0; i < 2*StateSize+1; ++i)
        for (int j = 0; j < StateSize; ++j)
            predDiff(i, j) -= predictedState.stateMean(j);
    auto obsDiff = obsPointsPredicted.points;
    for (int i = 0; i < 2*StateSize+1; ++i)
        for (int j = 0; j < ObservationSize; ++j)
            obsDiff(i, j) -= obsMomentsPredicted.stateMean(j);
    
    auto crossSigma = predDiff.transpose() * predictedPoints.weightsMean.asDiagonal() * obsDiff;

    // Kalman gain
    auto K = crossSigma * obsMomentsPredicted.stateCovariance.completeOrthogonalDecomposition().pseudoInverse();

    // Correct
    Moments<StateSize, T> momentsFiltered;
    momentsFiltered.stateMean = predictedState.stateMean + K * (observation - obsMomentsPredicted.stateMean);
    momentsFiltered.stateCovariance = predictedState.stateCovariance - K * crossSigma.transpose();
    
    return momentsFiltered;
}


} // namespace cppkalman
#endif // CPPKALMAN_IMPL
