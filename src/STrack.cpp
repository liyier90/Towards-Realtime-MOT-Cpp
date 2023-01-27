/*
author: samylee
github: https://github.com/samylee
date: 08/19/2021
*/

#include "STrack.h"

#include <vector>

#include <opencv2/opencv.hpp>

STrack::STrack(
    const std::vector<float> &rTlwh
  , float score
  , std::vector<float> features 
  , int bufferSize)
  : mTrackId {0},
    mIsActivated {false},
    mState {TrackState::New},
    mAlpha {0.9},
    mTlwhCache {rTlwh},
    mTlwh(4),
    mTlbr(4),
    mFrameId {0},
    mStartFrame {0},
    mTrackletLen {0},
    mScore {score}
{
  this->StaticTlwh();
  this->StaticTlbr();
  this->UpdateFeatures(features);
}

STrack::~STrack()
{}

void STrack::Activate(
    jde_kalman::KalmanFilter &rKalmanFilter
  , int frameId)
{
  mKalmanFilter = rKalmanFilter;
  mTrackId = this->NextId();

  std::vector<float> tlwh(4);
  tlwh[0] = mTlwhCache[0];
  tlwh[1] = mTlwhCache[1];
  tlwh[2] = mTlwhCache[2];
  tlwh[3] = mTlwhCache[3];

  auto xyah = this->TlwhToXyah(tlwh);

  DETECTBOX xyah_box;
  xyah_box[0] = xyah[0];
  xyah_box[1] = xyah[1];
  xyah_box[2] = xyah[2];
  xyah_box[3] = xyah[3];

  auto mc = mKalmanFilter.initiate(xyah_box);
  mMean = mc.first;
  mCovariance = mc.second;

  this->StaticTlwh();
  this->StaticTlbr();

  mTrackletLen = 0;
  mState = TrackState::Tracked;
  mFrameId = frameId;
  mStartFrame = frameId;
}

void STrack::ReActivate(
    STrack &rNewTrack
  , int frameId
  , bool newId)
{
  auto xyah = this->TlwhToXyah(rNewTrack.mTlwh);
  DETECTBOX xyah_box;
  xyah_box[0] = xyah[0];
  xyah_box[1] = xyah[1];
  xyah_box[2] = xyah[2];
  xyah_box[3] = xyah[3];
  auto mc = mKalmanFilter.update(mMean, mCovariance, xyah_box);
  mMean = mc.first;
  mCovariance = mc.second;

  this->StaticTlwh();
  this->StaticTlbr();

  this->UpdateFeatures(rNewTrack.mCurrFeat);
  mTrackletLen = 0;
  mState = TrackState::Tracked;
  mIsActivated = true;
  mFrameId = frameId;
  if (newId) {
    mTrackId = NextId();
  }
}

void STrack::Update(
    STrack &rNewTrack
  , int frameId
  , bool updateFeature)
{
  mFrameId = frameId;
  mTrackletLen++;

  auto xyah = this->TlwhToXyah(rNewTrack.mTlwh);
  DETECTBOX xyah_box;
  xyah_box[0] = xyah[0];
  xyah_box[1] = xyah[1];
  xyah_box[2] = xyah[2];
  xyah_box[3] = xyah[3];

  auto mc = mKalmanFilter.update(mMean, mCovariance, xyah_box);
  mMean = mc.first;
  mCovariance = mc.second;

  this->StaticTlwh();
  this->StaticTlbr();

  mState = TrackState::Tracked;
  mIsActivated = true;

  mScore = rNewTrack.mScore;
  if (updateFeature) {
    this->UpdateFeatures(rNewTrack.mCurrFeat);
  }
}

void STrack::StaticTlwh()
{
  if (mMean.isZero()) {
    mTlwh[0] = mTlwhCache[0];
    mTlwh[1] = mTlwhCache[1];
    mTlwh[2] = mTlwhCache[2];
    mTlwh[3] = mTlwhCache[3];
    return;
  }

  mTlwh[0] = mMean[0];
  mTlwh[1] = mMean[1];
  mTlwh[2] = mMean[2];
  mTlwh[3] = mMean[3];

  mTlwh[2] *= mTlwh[3];
  mTlwh[0] -= mTlwh[2] / 2;
  mTlwh[1] -= mTlwh[3] / 2;
}

void STrack::StaticTlbr()
{
  mTlbr.clear();
  mTlbr.assign(mTlwh.begin(), mTlwh.end());
  mTlbr[2] += mTlbr[0];
  mTlbr[3] += mTlbr[1];
}

std::vector<float> STrack::TlwhToXyah(const std::vector<float> &rTlwh)
{
  std::vector<float> tlwh_output = rTlwh;
  tlwh_output[0] += tlwh_output[2] / 2;
  tlwh_output[1] += tlwh_output[3] / 2;
  tlwh_output[2] /= tlwh_output[3];
  return tlwh_output;
}

std::vector<float> STrack::to_xyah()
{
  return this->TlwhToXyah(mTlwh);
}

std::vector<float> STrack::TlbrToTlwh(std::vector<float> &rTlbr)
{
  rTlbr[2] -= rTlbr[0];
  rTlbr[3] -= rTlbr[1];
  return rTlbr;
}

void STrack::mark_lost()
{
  mState = TrackState::Lost;
}

void STrack::MarkRemoved()
{
  mState = TrackState::Removed;
}

int STrack::NextId()
{
  static int _count = 0;
  _count++;
  return _count;
}

int STrack::EndFrame()
{
  return mFrameId;
}

void STrack::UpdateFeatures(std::vector<float> feat)
{
  cv::Mat feat_mat(feat);
  auto feat_value = cv::norm(feat_mat);
  for (int i = 0; i < feat.size(); ++i) {
    feat[i] /= feat_value;
  }
  mCurrFeat.assign(feat.begin(), feat.end());
  if (mSmoothFeat.size() == 0) {
    mSmoothFeat.assign(feat.begin(), feat.end());
  } else {
    for (int i = 0; i < mSmoothFeat.size(); ++i) {
      mSmoothFeat[i] = mAlpha * mSmoothFeat[i] + (1 - mAlpha) * feat[i];
    }
  }

  cv::Mat smooth_feat_mat(mSmoothFeat);
  auto smooth_feat_value = norm(smooth_feat_mat);
  for (int i = 0; i < mSmoothFeat.size(); ++i) {
    mSmoothFeat[i] /= smooth_feat_value;
  }
}

void STrack::MultiPredict(
    std::vector<STrack*> &rStracks
  , jde_kalman::KalmanFilter &rKalmanFilter)
{
  for (int i = 0; i < rStracks.size(); ++i) {
    if (rStracks[i]->mState != TrackState::Tracked) {
      rStracks[i]->mMean[7] = 0;
    }
    rKalmanFilter.predict(rStracks[i]->mMean, rStracks[i]->mCovariance);
    rStracks[i]->StaticTlwh();
    rStracks[i]->StaticTlbr();
  }
}
