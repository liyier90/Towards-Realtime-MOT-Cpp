#ifndef DATATYPE_H_
#define DATATYPE_H_

#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DetectBox;
typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DetectBoxes;
typedef Eigen::Matrix<float, Eigen::Dynamic, 128, Eigen::RowMajor> Features;

// KalmanFilter
typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> KalmanMean;
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KalmanCov;
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> KalmanHMean;
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> KalmanHCov;
using KalmanData = std::pair<KalmanMean, KalmanCov>;
using KalmanHData = std::pair<KalmanHMean, KalmanHCov>;

// main
// using ResultData = std::pair<int, DetectBox>;

// tracker:
// using TrackerData = std::pair<int, Features>;
// using MatchData = std::pair<int, int>;
// typedef struct TrackerMatchData {
//   std::vector<MatchData> matches;
//   std::vector<int> unmatched_tracks;
//   std::vector<int> unmatched_detections;
// } TrackerMatchData;

// LinearAssignment:
// typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> DynamicMat;

#endif  // DATATYPE_H_

