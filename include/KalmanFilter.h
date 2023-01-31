#ifndef KALMANFILTER_H_
#define KALMANFILTER_H_

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataType.h"

namespace jde_kalman
{
	class KalmanFilter
	{
	public:
		static const double chi2inv95[10];
		KalmanFilter();
		KalmanData initiate(const DetectBox& measurement);
		void predict(KalmanMean& mean, KalmanCov& covariance);

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
}

#endif  // KALMANFILTER_H_
 
