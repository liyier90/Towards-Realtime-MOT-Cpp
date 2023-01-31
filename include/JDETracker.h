#pragma once

#include <cfloat>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "STrack.h"

class JDETracker
{
public:
	JDETracker(
      const std::string &rModelPath
    , int frameRate = 30
    , int trackBuffer = 30);
	~JDETracker();

	void Update(cv::Mat image);

	int mNetWidth;
	int mNetHeight;

private:
  cv::Mat Preprocess(cv::Mat image);

	torch::jit::script::Module mModel;
	torch::Device *mpDevice;

	float mScoreThreshold;
	float mNmsThreshold;
	int mFrameId;
	int mMaxTimeLost;

	std::vector<STrack> mTrackedStracks;
	std::vector<STrack> mLostStracks;
	std::vector<STrack> mRemovedStracks;
	jde_kalman::KalmanFilter mKalmanFilter;
};

namespace strack_util
{
std::vector<STrack*> CombineStracks(
    std::vector<STrack*> &rStracks1
  , std::vector<STrack> &rStracks2);

std::vector<STrack> CombineStracks(
    std::vector<STrack> &rStracks1
  , std::vector<STrack> &rStracks2);

void RemoveDuplicateStracks(
    const std::vector<STrack> &rStracks1
  , const std::vector<STrack> &rStracks2
  , std::vector<STrack> &rRes1
  , std::vector<STrack> &rRes2);

std::vector<STrack> SubstractStracks(
    std::vector<STrack> &rStracks1
  , std::vector<STrack> &rStracks2);
}  // namespace  strack_util
