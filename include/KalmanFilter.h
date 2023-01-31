#ifndef INCLUDE_KALMANFILTER_H_
#define INCLUDE_KALMANFILTER_H_

#include <vector>

#include <Eigen/Core>  // NOLINT
#include <Eigen/Dense>  // NOLINT

#include "DataType.h"

namespace jde_kalman {
class KalmanFilter {
 public:
  static const double chi2inv95[10];

  KalmanFilter();

  KalmanData Initiate(const DetectBox &rMeasurement) const;

  void Predict(
      KalmanMean *pMean,
      KalmanCov *pCovariance) const;

  KalmanHData Project(
      const KalmanMean &rMean,
      const KalmanCov &rCovariance) const;

  KalmanData Update(
      const KalmanMean &rMean,
      const KalmanCov &rCovariance,
      const DetectBox &rMeasurement);

  Eigen::Matrix<float, 1, -1> GatingDistance(
      const KalmanMean &rMean,
      const KalmanCov &rCovariance,
      const std::vector<DetectBox> &rMeasurements,
      bool onlyPosition = false) const;

 private:
  static constexpr int mkNDim = 4;
  static constexpr double mkDt = 1.0;
  Eigen::Matrix<float, 8, 8, Eigen::RowMajor> mMotionMat;
  Eigen::Matrix<float, 4, 8, Eigen::RowMajor> mUpdateMat;
  float mStdWeightPosition;
  float mStdWeightVelocity;
};
}  // namespace jde_kalman

#endif  // INCLUDE_KALMANFILTER_H_

