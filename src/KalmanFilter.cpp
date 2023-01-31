#include "KalmanFilter.h"

#include <Eigen/Cholesky>

namespace jde_kalman {
const double KalmanFilter::chi2inv95[10] = {
    0,
    3.8415,
    5.9915,
    7.8147,
    9.4877,
    11.070,
    12.592,
    14.067,
    15.507,
    16.919
};

KalmanFilter::KalmanFilter()
  : mMotionMat {Eigen::MatrixXf::Identity(8, 8)},
    mUpdateMat {Eigen::MatrixXf::Identity(4, 8)},
    mStdWeightPosition {1.0 / 20},
    mStdWeightVelocity {1.0 / 160}
{
  for (int i = 0; i < mkNDim; ++i) {
    mMotionMat(i, mkNDim + i) = mkDt;
  }
}

KalmanData KalmanFilter::Initiate(const DetectBox &rMeasurement) const {
  auto mean_pos = rMeasurement;
  DetectBox mean_vel;
  for (int i = 0; i < 4; i++) {
    mean_vel[i] = 0;
  }

  KalmanMean mean;
  for (int i = 0; i < 8; i++) {
    if (i < 4) {
      mean[i] = mean_pos[i];
    } else {
      mean[i] = mean_vel[i - 4];
    }
  }

  KalmanMean std_dev;
  std_dev <<
      2 * mStdWeightPosition * rMeasurement[3],
      2 * mStdWeightPosition * rMeasurement[3],
      1e-2,
      2 * mStdWeightPosition * rMeasurement[3],
      10 * mStdWeightVelocity * rMeasurement[3],
      10 * mStdWeightVelocity * rMeasurement[3],
      1e-5,
      10 * mStdWeightVelocity * rMeasurement[3];

  KalmanMean tmp = std_dev.array().square();
  KalmanCov var = tmp.asDiagonal();
  return std::make_pair(mean, var);
}

void KalmanFilter::Predict(
    KalmanMean *pMean,
    KalmanCov *pCovariance) const {
  // revise the data;
  DetectBox std_pos;
  std_pos <<
      mStdWeightPosition * (*pMean)[3],
      mStdWeightPosition * (*pMean)[3],
      1e-2,
      mStdWeightPosition * (*pMean)[3];
  DetectBox std_vel;
  std_vel <<
      mStdWeightVelocity * (*pMean)[3],
      mStdWeightVelocity * (*pMean)[3],
      1e-5,
      mStdWeightVelocity * (*pMean)[3];
  KalmanMean tmp;
  tmp.block<1, 4>(0, 0) = std_pos;
  tmp.block<1, 4>(0, 4) = std_vel;
  tmp = tmp.array().square();
  KalmanCov motion_cov = tmp.asDiagonal();
  KalmanMean mean = mMotionMat * pMean->transpose();
  KalmanCov covariance = mMotionMat * (*pCovariance) * mMotionMat.transpose();
  covariance += motion_cov;

  *pMean = mean;
  *pCovariance = covariance;
}

KalmanHData KalmanFilter::Project(
    const KalmanMean &rMean,
    const KalmanCov &rCovariance) const {
  DetectBox std_dev;
  std_dev <<
      mStdWeightPosition * rMean[3],
      mStdWeightPosition * rMean[3],
      1e-1,
      mStdWeightPosition * rMean[3];
  KalmanHMean mean = mUpdateMat * rMean.transpose();
  KalmanHCov covariance = mUpdateMat * rCovariance * mUpdateMat.transpose();
  Eigen::Matrix<float, 4, 4> diag = std_dev.asDiagonal();
  diag = diag.array().square().matrix();
  covariance += diag;
  return std::make_pair(mean, covariance);
}

KalmanData KalmanFilter::Update(
    const KalmanMean &rMean,
    const KalmanCov &rCovariance,
    const DetectBox &rMeasurement) {
  auto pa = this->Project(rMean, rCovariance);
  KalmanHMean projected_mean = pa.first;
  KalmanHCov projected_cov = pa.second;

  // chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True,
  //     check_finite=False)
  // kalman_gain = scipy.linalg.cho_solve((cho_factor, lower),
  //     np.dot(covariance, self.mUpdateMat.T).T, check_finite=False).T
  Eigen::Matrix<float, 4, 8> B = (rCovariance * mUpdateMat.transpose())
      .transpose();
  Eigen::Matrix<float, 8, 4> kalman_gain = projected_cov.llt().solve(B)
      .transpose();
  Eigen::Matrix<float, 1, 4> innovation = rMeasurement - projected_mean;
  auto tmp = innovation * kalman_gain.transpose();
  KalmanMean mean = (rMean.array() + tmp.array()).matrix();
  KalmanCov covariance = rCovariance - kalman_gain * projected_cov *
      kalman_gain.transpose();
  return std::make_pair(mean, covariance);
}

Eigen::Matrix<float, 1, -1> KalmanFilter::GatingDistance(
    const KalmanMean &rMean,
    const KalmanCov &rCovariance,
    const std::vector<DetectBox> &rMeasurements,
    bool onlyPosition) const {
  auto pa = this->Project(rMean, rCovariance);
  if (onlyPosition) {
    printf("not implement!");
    exit(0);
  }
  KalmanHMean mean1 = pa.first;
  KalmanHCov covariance1 = pa.second;

  // Eigen::Matrix<float, -1, 4, Eigen::RowMajor> d(size, 4);
  DetectBoxes d(rMeasurements.size(), 4);
  int pos = 0;
  for (DetectBox box : rMeasurements) {
    d.row(pos++) = box - mean1;
  }
  Eigen::Matrix<float, -1, -1, Eigen::RowMajor> factor = covariance1.llt()
      .matrixL();
  Eigen::Matrix<float, -1, -1> z = factor.triangularView<Eigen::Lower>()
      .solve<Eigen::OnTheRight>(d)
      .transpose();
  auto zz = ((z.array()) * (z.array())).matrix();
  auto square_maha = zz.colwise().sum();
  return square_maha;
}
}  // namespace jde_kalman
