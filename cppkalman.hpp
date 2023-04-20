/*
*/


#ifndef CPPKALMAN_H
#define CPPKALMAN_H

#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include <functional>
#include <optional>
#include <utility>

namespace cppkalman {

// using sizetype = size_t;
// Eigen uses int for its template sizes for some reason
// This is a hack to avoid -Wconversion warnings (Assume you don't need a vector bigger than 2 billion)
using sizetype = int;

template<sizetype N, typename T>
struct Moments
{
    Eigen::Vector<T, N> stateMean;
    Eigen::Matrix<T, N, N> stateCovariance;
};

template<sizetype StateSize, sizetype N, typename T>
struct SigmaPoints
{
    Eigen::Matrix<T, 2*StateSize+1, N> points;
    Eigen::Vector<T, 2*StateSize+1> weightsMean;
    Eigen::Vector<T, 2*StateSize+1> weightsCovariance;
};

template<sizetype StateSize, sizetype ObservationSize, typename T=double>
class AdditiveUnscentedKalmanFilter
{
public:
    using StateCovariance = Eigen::Matrix<T, StateSize, StateSize>;
    using State = Eigen::Vector<T, StateSize>;
    using Observation = Eigen::Vector<T, ObservationSize>;

    using TransitionFunction = std::function<State(State)>;
    using ObservationFunction = std::function<Observation(State)>;

    AdditiveUnscentedKalmanFilter(TransitionFunction&&, ObservationFunction&&);

    [[nodiscard]] std::pair<Moments<StateSize, T>, SigmaPoints<StateSize, StateSize, T>> Predict(const Moments<StateSize, T>&);
    [[nodiscard]] Moments<StateSize, T> Update(const Moments<StateSize, T>&, const SigmaPoints<StateSize, StateSize, T>&, const std::optional<Observation>&);

public:
    TransitionFunction mTransitionFunction;
    ObservationFunction mObservationFunction;
};


} // namespace cppkalman
#endif // CPPKALMAN_H



#ifdef CPPKALMAN_IMPL
namespace cppkalman {


template<sizetype StateSize, typename T>
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
    for (sizetype i = 0; i < StateSize; ++i)
        for (sizetype j = i+1; j < StateSize; ++j)
            sigma2(i, j) = 0;

    const T lambda = (alpha * alpha) * (StateSize + kappa) - StateSize;
    const T c = StateSize + lambda;

    for (sizetype i = 0; i < StateSize; ++i) {
        for (sizetype j = 0; j < 2*StateSize+1; ++j)
            points.points(j, i) = mu(i);
        for (sizetype j = 1; j < StateSize + 1; ++j)
            points.points(j, i) += sigma2(i, j-1) * (T) sqrt(c);
        for (sizetype j = StateSize + 1; j < 2*StateSize+1; ++j)
            points.points(j, i) -= sigma2(i, j-(StateSize+1)) * (T) sqrt(c);
    }

    points.weightsMean.setOnes();
    points.weightsMean(0) = lambda / c;
    for (sizetype i = 1; i < 2*StateSize+1; ++i)
        points.weightsMean(i) = (T) 0.5 / c;

    points.weightsCovariance = points.weightsMean;
    points.weightsCovariance(0) = lambda / c + (1 - alpha * alpha + beta);

    return points;
}


template<sizetype StateSize, sizetype ObservationSize, typename T>
Moments<ObservationSize, T>
static
Points2Moments(const SigmaPoints<StateSize, ObservationSize, T>& sigmaPoints)
{
    Moments<ObservationSize, T> moments;
    const auto& [points, weightsMean, weightsCovariance] = sigmaPoints;
    
    moments.stateMean = points.transpose() * weightsMean;
    Eigen::Matrix<T, ObservationSize, 2*StateSize+1> pointsDiff = points.transpose();
    for (sizetype i = 0; i < ObservationSize; ++i)
        for (sizetype j = 0; j < 2*StateSize+1; ++j)
            pointsDiff(i, j) -= moments.stateMean(i);


    moments.stateCovariance.setIdentity();
    moments.stateCovariance += pointsDiff * weightsCovariance.asDiagonal() * pointsDiff.transpose();

    return moments;
}


template<sizetype StateSize, sizetype ObservationSize, typename T, typename Function>
std::pair<SigmaPoints<StateSize, ObservationSize, T>, Moments<ObservationSize, T>>
static
UnscentedTransform(const SigmaPoints<StateSize, StateSize, T>& sigmaPoints, Function f) // FIXME: constrain function with concept
{
    SigmaPoints<StateSize, ObservationSize, T> points;
    points.weightsCovariance = sigmaPoints.weightsCovariance;
    points.weightsMean = sigmaPoints.weightsMean;
    for (sizetype i = 0; i < 2*StateSize+1; ++i) {
        Eigen::Vector<T, ObservationSize> row = f(sigmaPoints.points.row(i));
        for (sizetype j = 0; j < ObservationSize; ++j)
            points.points(i, j) = row(j);
    }

    return std::make_pair(points, Points2Moments(points));
}


template<sizetype StateSize, sizetype ObservationSize, typename T>
AdditiveUnscentedKalmanFilter<StateSize, ObservationSize, T>::AdditiveUnscentedKalmanFilter(
    TransitionFunction&& transitionFunction,
    ObservationFunction&& observationFunction
)
    : mTransitionFunction{transitionFunction}
    , mObservationFunction{observationFunction}
{
}


template<sizetype StateSize, sizetype ObservationSize, typename T>
std::pair<Moments<StateSize, T>, SigmaPoints<StateSize, StateSize, T>>
AdditiveUnscentedKalmanFilter<StateSize, ObservationSize, T>::Predict(
    const Moments<StateSize, T>& momentsState)
{
    SigmaPoints pointsState = Moments2Points(momentsState);
    auto [_, moments] = UnscentedTransform<StateSize, StateSize, T>(pointsState, mTransitionFunction);
    return std::make_pair(moments, Moments2Points(moments));
}

template<sizetype StateSize, sizetype ObservationSize, typename T>
Moments<StateSize, T>
AdditiveUnscentedKalmanFilter<StateSize, ObservationSize, T>::Update(
    const Moments<StateSize, T>& momentsPredicted,
    const SigmaPoints<StateSize, StateSize, T>& pointsPredicted,
    const std::optional<Observation>& observation)
{
    if (!observation)
        return momentsPredicted;

    Moments<StateSize, T> momentsFiltered;
    auto [obsPointsPredicted, obsMomentsPredicted] = UnscentedTransform<StateSize, ObservationSize, T>(pointsPredicted, mObservationFunction);

    auto predDiff = pointsPredicted.points;
    for (sizetype i = 0; i < 2*StateSize+1; ++i)
        for (sizetype j = 0; j < StateSize; ++j)
            predDiff(i, j) -= momentsPredicted.stateMean(j);
    auto obsDiff = obsPointsPredicted.points;
    for (sizetype i = 0; i < 2*StateSize+1; ++i)
        for (sizetype j = 0; j < ObservationSize; ++j)
            obsDiff(i, j) -= obsMomentsPredicted.stateMean(j);
    
    auto crossSigma = predDiff.transpose() * pointsPredicted.weightsMean.asDiagonal() * obsDiff;

    // Kalman gain
    auto K = crossSigma * obsMomentsPredicted.stateCovariance.completeOrthogonalDecomposition().pseudoInverse();
    // Correct
    momentsFiltered.stateMean = momentsPredicted.stateMean + K * (*observation - obsMomentsPredicted.stateMean);
    momentsFiltered.stateCovariance = momentsPredicted.stateCovariance - K * crossSigma.transpose();
    
    return momentsFiltered;
}


} // namespace cppkalman
#endif // CPPKALMAN_IMPL
