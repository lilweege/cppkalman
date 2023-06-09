#include "example_templates.hpp"

#define CPPKALMAN_IMPL
#include "cppkalman.hpp"

// Forward declare templates
namespace cppkalman {

template struct Moments<STATE_SIZE, FloatType>;
template struct SigmaPoints<STATE_SIZE, STATE_SIZE, FloatType>;
template AdditiveUnscentedKalmanFilter<STATE_SIZE, OBSERVATION_SIZE, FloatType>::AdditiveUnscentedKalmanFilter(TransitionFunction&&, ObservationFunction&&);
template std::pair<Moments<STATE_SIZE, FloatType>, SigmaPoints<STATE_SIZE, STATE_SIZE, FloatType>> AdditiveUnscentedKalmanFilter<STATE_SIZE, OBSERVATION_SIZE, FloatType>::Predict(const Moments<STATE_SIZE, FloatType>&, const Eigen::Matrix<FloatType, STATE_SIZE, STATE_SIZE>&);
template Moments<STATE_SIZE, FloatType> AdditiveUnscentedKalmanFilter<STATE_SIZE, OBSERVATION_SIZE, FloatType>::Update(const Moments<STATE_SIZE, FloatType>&, const SigmaPoints<STATE_SIZE, STATE_SIZE, FloatType>&, const Eigen::Matrix<FloatType, OBSERVATION_SIZE, OBSERVATION_SIZE>&, const AdditiveUnscentedKalmanFilter<STATE_SIZE, OBSERVATION_SIZE, FloatType>::Observation&);

} // namespace cppkalman