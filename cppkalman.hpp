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

template<size_t N, typename T>
struct Moments
{
    Eigen::Vector<T, N> stateMean;
    Eigen::Matrix<T, N, N> stateCovariance;
};

template<size_t StateSize, size_t N, typename T>
struct SigmaPoints
{
    Eigen::Matrix<T, 2*StateSize+1, N> points;
    Eigen::Vector<T, 2*StateSize+1> weightsMean;
    Eigen::Vector<T, 2*StateSize+1> weightsCovariance;
};

template<size_t StateSize, size_t ObservationSize, typename T=double>
class AdditiveUnscentedKalmanFilter
{
public:
    using StateCovariance = Eigen::Matrix<T, StateSize, StateSize>;
    using State = Eigen::Vector<T, StateSize>;
    using Observation = Eigen::Vector<T, ObservationSize>;

    using TransitionFunction = std::function<State(State)>;
    using ObservationFunction = std::function<Observation(State)>;

    AdditiveUnscentedKalmanFilter(TransitionFunction&&, ObservationFunction&&);

    void Predict(const Moments<StateSize, T>&);
    Moments<StateSize, T> Update(const std::optional<Observation>&);

public:
    const TransitionFunction mTransitionFunction;
    const ObservationFunction mObservationFunction;

    Moments<StateSize, T> mMomentsPredicted;
    SigmaPoints<StateSize, StateSize, T> mPointsPredicted;
};


} // namespace cppkalman
#endif // CPPKALMAN_H



#ifdef CPPKALMAN_IMPL
namespace cppkalman {


template<size_t StateSize, typename T>
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
    for (size_t i = 0; i < StateSize; ++i)
        for (size_t j = i+1; j < StateSize; ++j)
            sigma2(i, j) = 0;

    const T lambda = (alpha * alpha) * (StateSize + kappa) - StateSize;
    const T c = StateSize + lambda;

    for (size_t i = 0; i < StateSize; ++i) {
        for (size_t j = 0; j < 2*StateSize+1; ++j)
            points.points(j, i) = mu(i);
        for (size_t j = 1; j < StateSize + 1; ++j)
            points.points(j, i) += sigma2(i, j-1) * sqrt(c);
        for (size_t j = StateSize + 1; j < 2*StateSize+1; ++j)
            points.points(j, i) -= sigma2(i, j-(StateSize+1)) * sqrt(c);
    }

    points.weightsMean.setOnes();
    points.weightsMean(0) = lambda / c;
    for (size_t i = 1; i < 2*StateSize+1; ++i)
        points.weightsMean(i) = 0.5 / c;

    points.weightsCovariance = points.weightsMean;
    points.weightsCovariance(0) = lambda / c + (1 - alpha * alpha + beta);

    return points;
}


template<size_t StateSize, size_t ObservationSize, typename T>
Moments<ObservationSize, T>
static
Points2Moments(const SigmaPoints<StateSize, ObservationSize, T>& sigmaPoints)
{
    Moments<ObservationSize, T> moments;
    const auto& [points, weightsMean, weightsCovariance] = sigmaPoints;
    
    moments.stateMean = points.transpose() * weightsMean;
    Eigen::Matrix<T, ObservationSize, 2*StateSize+1> pointsDiff = points.transpose();
    for (size_t i = 0; i < ObservationSize; ++i)
        for (size_t j = 0; j < 2*StateSize+1; ++j)
            pointsDiff(i, j) -= moments.stateMean(i);


    moments.stateCovariance.setIdentity();
    moments.stateCovariance += pointsDiff * weightsCovariance.asDiagonal() * pointsDiff.transpose();

    return moments;
}


template<size_t StateSize, size_t ObservationSize, typename T>
std::pair<SigmaPoints<StateSize, ObservationSize, T>, Moments<ObservationSize, T>>
static
UnscentedTransform(const SigmaPoints<StateSize, StateSize, T>& sigmaPoints, auto f) // FIXME: constrain function with concept
{
    SigmaPoints<StateSize, ObservationSize, T> points;
    points.weightsCovariance = sigmaPoints.weightsCovariance;
    points.weightsMean = sigmaPoints.weightsMean;
    for (size_t i = 0; i < 2*StateSize+1; ++i) {
        Eigen::Vector<T, ObservationSize> row = f(sigmaPoints.points.row(i));
        for (size_t j = 0; j < row.size(); ++j)
            points.points(i, j) = row(j);
    }

    return std::make_pair(points, Points2Moments(points));
}


template<size_t StateSize, size_t ObservationSize, typename T>
AdditiveUnscentedKalmanFilter<StateSize, ObservationSize, T>::AdditiveUnscentedKalmanFilter(TransitionFunction&& transitionFunction, ObservationFunction&& observationFunction)
    : mTransitionFunction{transitionFunction}
    , mObservationFunction{observationFunction}
{
}


template<size_t StateSize, size_t ObservationSize, typename T>
void
AdditiveUnscentedKalmanFilter<StateSize, ObservationSize, T>::Predict(const Moments<StateSize, T>& momentsState)
{
    SigmaPoints pointsState = Moments2Points(momentsState);
    auto [_, moments] = UnscentedTransform<StateSize, StateSize, T>(pointsState, mTransitionFunction);

    mMomentsPredicted = std::move(moments);
    mPointsPredicted = Moments2Points(mMomentsPredicted);
}

template<size_t StateSize, size_t ObservationSize, typename T>
Moments<StateSize, T>
AdditiveUnscentedKalmanFilter<StateSize, ObservationSize, T>::Update(const std::optional<Observation>& observation)
{
    if (!observation)
        return mMomentsPredicted;

    Moments<StateSize, T> momentsFiltered;
    auto [obsPointsPredicted, obsMomentsPredicted] = UnscentedTransform<StateSize, ObservationSize, T>(mPointsPredicted, mObservationFunction);

    auto x = mPointsPredicted.points;
    for (size_t i = 0; i < 2*StateSize+1; ++i)
        for (size_t j = 0; j < StateSize; ++j)
            x(i, j) -= mMomentsPredicted.stateMean(j);
    auto y = obsPointsPredicted.points;
    for (size_t i = 0; i < 2*StateSize+1; ++i)
        for (size_t j = 0; j < StateSize; ++j)
            y(i, j) -= obsMomentsPredicted.stateMean(j);
    
    auto crossSigma = x.transpose() * mPointsPredicted.weightsMean.asDiagonal() * y;

    // Kalman gain
    auto K = crossSigma * obsMomentsPredicted.stateCovariance.completeOrthogonalDecomposition().pseudoInverse();
    // Correct
    momentsFiltered.stateMean = mMomentsPredicted.stateMean + K * (*observation - obsMomentsPredicted.stateMean);
    momentsFiltered.stateCovariance = mMomentsPredicted.stateCovariance - K * crossSigma.transpose();
    
    return momentsFiltered;
}


} // namespace cppkalman
#endif // CPPKALMAN_IMPL
