/*
author: samylee
github: https://github.com/samylee
date: 08/19/2021
*/

#pragma once

#include <cfloat>
#include <string>
#include <vector>

#include "STrack.h"

class JDETracker
{
public:
	JDETracker(
      const std::string &rModelPath
    , int frameRate = 30
    , int trackBuffer = 30);
	~JDETracker();

	void Update(const std::string &rVideoPath);

private:
  cv::Size GetSize(
      int vw
    , int vh
    , int dw
    , int dh);
	Mat Letterbox(
      Mat img
    , int height
    , int width);
	torch::Tensor NonMaxSuppression(torch::Tensor prediction);
	torch::Tensor xywh2xyxy(torch::Tensor x);
	torch::Tensor Nms(
      const torch::Tensor &boxes
    , const torch::Tensor &scores
    , float overlap);
	void ScaleCoords(
      torch::Tensor &coords
    , Size img_size
    , Size img0_shape);

	std::vector<STrack*> CombineStracks(
      std::vector<STrack*> &rStracks1
    , std::vector<STrack> &rStracks2);
	std::vector<STrack> CombineStracks(
      std::vector<STrack> &rStracks1
    , std::vector<STrack> &rStracks2);

	std::vector<STrack> SubstractStracks(
      std::vector<STrack> &rStracks1
    , std::vector<STrack> &rStracks2);
	void RemoveDuplicateStracks(
      const std::vector<STrack> &rStracks1
    , const std::vector<STrack> &rStracks2
    , std::vector<STrack> &rRes1
    , std::vector<STrack> &rRes2);

	void EmbeddingDistance(
      std::vector<STrack*> &tracks
    , std::vector<STrack> &detections
    , std::vector<std::vector<float>> &cost_matrix
    , int *cost_matrix_size
    , int *cost_matrix_size_size);
	void FuseMotion(
      std::vector<std::vector<float>> &rCostMatrix
    , std::vector<STrack*> &rTracks
    , std::vector<STrack> &rDetections
    , bool onlyPosition = false
    , float coeff = 0.98);

	void LinearAssignment(
      std::vector<std::vector<float>> &rCostMatrix
    , int numRows 
    , int numCols
    , float threshhold
    , std::vector<std::vector<int>> &rMatches
    , std::vector<int> &rUnmatched1
    , std::vector<int> &rUnmatched2);

	std::vector<std::vector<float>> IouDistance(
      const std::vector<STrack*> &rTracks1
    , const std::vector<STrack> &rTracks2
    , int &rNumRows
    , int &rNumCols);

	std::vector<std::vector<float>> IouDistance(
      const std::vector<STrack> &rTracks1
    , const std::vector<STrack> &rTracks2);

	std::vector<std::vector<float>> Ious(
      std::vector<std::vector<float>> &atlbrs
    , std::vector<std::vector<float>> &btlbrs);

	double lapjv(
      const std::vector<std::vector<float>> &rCostMatrix
    , std::vector<int> &rRowsol
    , std::vector<int> &rColsol
    , bool extendCost = false
    , float costLimit = FLT_MAX
    , bool returnCost = true);

	Scalar get_color(int idx);

	torch::jit::script::Module jde_model;
	torch::Device *mpDevice;

	int mNetWidth;
	int mNetHeight;
	float mScoreThreshold;
	float mNmsThreshold;
	int mFrameId;
	int mMaxTimeLost;

	std::vector<STrack> mTrackedStracks;
	std::vector<STrack> mLostStracks;
	std::vector<STrack> mRemovedStracks;
	jde_kalman::KalmanFilter mKalmanFilter;
};
