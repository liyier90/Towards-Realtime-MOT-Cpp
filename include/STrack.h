/*
author: samylee
github: https://github.com/samylee
date: 08/19/2021
*/

#pragma once

#include <vector>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

#include "kalmanFilter.h"

using namespace cv;

enum TrackState { New = 0, Tracked, Lost, Removed };

class STrack
{
public:
	STrack(
      const std::vector<float> &rTlwh
    , float score
    , std::vector<float> features
    , int bufferSize = 30);

	~STrack();

	static std::vector<float> TlbrToTlwh(std::vector<float> &rTlbr);
	static void MultiPredict(
      std::vector<STrack*> &rStracks
    , jde_kalman::KalmanFilter &rKalmanFilter);

	void StaticTlwh();
	void StaticTlbr();

  std::vector<float> TlwhToXyah(const std::vector<float> &rTlwh);
  std::vector<float> to_xyah();

	void mark_lost();
	void MarkRemoved();
	int NextId();
	int EndFrame();
	
	void Activate(
      jde_kalman::KalmanFilter &rKalmanFilter
    , int frameId);
	void ReActivate(
      STrack &rNewTrack
    , int frameId
    , bool newId = false);
	void Update(
      STrack &rNewTrack
    , int frameId
    , bool updateFeature = true);

  int mTrackId;
	bool mIsActivated;
	int mState;

  std::vector<float> mCurrFeat;
  std::vector<float> mSmoothFeat;
	float mAlpha;

  std::vector<float> mTlwhCache;
  std::vector<float> mTlwh;
  std::vector<float> mTlbr;
  float mScore;

	int mFrameId;
	int mStartFrame;
  int mTrackletLen;

	KAL_MEAN mMean;
	KAL_COVA mCovariance;

private:
	void UpdateFeatures(std::vector<float> feat);
	jde_kalman::KalmanFilter mKalmanFilter;
};
