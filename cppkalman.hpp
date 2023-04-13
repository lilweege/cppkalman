/*
*/


#ifndef CPPKALMAN_H
#define CPPKALMAN_H

#include <Eigen/Dense>
#include <functional>
#include <optional>

namespace cppkalman {

template<size_t StateSize, size_t ObservationSize, typename T=double>
class AdditiveUnscentedKalmanFilter
{
public:
    using StateCovariance = Eigen::Matrix<T, StateSize, StateSize>;
    using State = Eigen::Vector<T, StateSize>;
    using Observation = Eigen::Vector<T, ObservationSize>;

    using TransitionFunction = std::function<State(State)>;
    using ObservationFunction = std::function<Observation(State)>;

    struct Sample {
        State filteredStateMean;
        StateCovariance filteredStateCovariance;
    };

    AdditiveUnscentedKalmanFilter(TransitionFunction&&, ObservationFunction&&);

    void Predict(Sample);
    Sample Update(std::optional<Observation>);

private:
    TransitionFunction mTransitionFunction;
    ObservationFunction mObservationFunction;

    T mMomentsPredicted;
    T mPointsPredicted;
};


} // namespace cppkalman
#endif // CPPKALMAN_H



#ifdef CPPKALMAN_IMPL
namespace cppkalman {


template<size_t StateSize, size_t ObservationSize, typename T>
AdditiveUnscentedKalmanFilter<StateSize, ObservationSize, T>::AdditiveUnscentedKalmanFilter(TransitionFunction&& transitionFunction, ObservationFunction&& observationFunction)
    : mTransitionFunction{transitionFunction}
    , mObservationFunction{observationFunction}
{
    
}

template<size_t StateSize, size_t ObservationSize, typename T>
void
AdditiveUnscentedKalmanFilter<StateSize, ObservationSize, T>::Predict(Sample currentSample)
{

}

template<size_t StateSize, size_t ObservationSize, typename T>
AdditiveUnscentedKalmanFilter<StateSize, ObservationSize, T>::Sample
AdditiveUnscentedKalmanFilter<StateSize, ObservationSize, T>::Update(std::optional<Observation> observation)
{
    return {};
}


} // namespace cppkalman
#endif // CPPKALMAN_IMPL
