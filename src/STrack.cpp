#include "STrack.h"

#include <vector>

#include <Eigen/Core>  // NOLINT
#include <Eigen/Dense>  // NOLINT
#include <opencv2/opencv.hpp>

STrack::STrack(
    std::vector<float> *pTlwh,
    float score,
    std::vector<float> features,
    int bufferSize)
  : mTrackId {0},
    mIsActivated {false},
    mState {TrackState::New},
    mAlpha {0.9},
    mTlwhCache {*pTlwh},
    mTlwh(4),
    mTlbr(4),
    mScore {score},
    mFrameId {0},
    mStartFrame {0},
    mTrackletLen {0},
    mMean {Eigen::Matrix<float, 1, 8, Eigen::RowMajor>::Zero()},
    mCovariance {Eigen::Matrix<float, 8, 8, Eigen::RowMajor>::Zero()},
    mpKalmanFilter {nullptr}
{
  this->StaticTlwh();
  this->StaticTlbr();
  this->UpdateFeatures(features);
}

STrack::~STrack()
{}

std::vector<float>* STrack::pTlbrToTlwh(std::vector<float> *pTlbr) {
  (*pTlbr)[2] -= (*pTlbr)[0];
  (*pTlbr)[3] -= (*pTlbr)[1];
  return pTlbr;
}

void STrack::MultiPredict(
    const std::vector<STrack*> &rStracks,
    const jde_kalman::KalmanFilter &rKalmanFilter) {
  for (int i = 0; i < rStracks.size(); ++i) {
    if (rStracks[i]->mState != TrackState::Tracked) {
      rStracks[i]->mMean[7] = 0;
    }
    rKalmanFilter.Predict(&(rStracks[i]->mMean), &(rStracks[i]->mCovariance));
    rStracks[i]->StaticTlwh();
    rStracks[i]->StaticTlbr();
  }
}

void STrack::StaticTlwh() {
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

void STrack::StaticTlbr() {
  mTlbr.clear();
  mTlbr.assign(mTlwh.begin(), mTlwh.end());
  mTlbr[2] += mTlbr[0];
  mTlbr[3] += mTlbr[1];
}

std::vector<float> STrack::TlwhToXyah(const std::vector<float> &rTlwh) const {
  std::vector<float> tlwh_output = rTlwh;
  tlwh_output[0] += tlwh_output[2] / 2;
  tlwh_output[1] += tlwh_output[3] / 2;
  tlwh_output[2] /= tlwh_output[3];
  return tlwh_output;
}

std::vector<float> STrack::ToXyah() const {
  return this->TlwhToXyah(mTlwh);
}

void STrack::MarkLost() {
  mState = TrackState::Lost;
}

void STrack::MarkRemoved() {
  mState = TrackState::Removed;
}

int STrack::NextId() {
  static int _count = 0;
  ++_count;
  return _count;
}

int STrack::EndFrame() {
  return mFrameId;
}

void STrack::Activate(
    jde_kalman::KalmanFilter *pKalmanFilter,
    int frameId) {
  mpKalmanFilter = pKalmanFilter;
  mTrackId = this->NextId();

  std::vector<float> tlwh(4);
  tlwh[0] = mTlwhCache[0];
  tlwh[1] = mTlwhCache[1];
  tlwh[2] = mTlwhCache[2];
  tlwh[3] = mTlwhCache[3];

  auto xyah = this->TlwhToXyah(tlwh);

  DetectBox xyah_box;
  xyah_box[0] = xyah[0];
  xyah_box[1] = xyah[1];
  xyah_box[2] = xyah[2];
  xyah_box[3] = xyah[3];

  auto mc = mpKalmanFilter->Initiate(xyah_box);
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
    const STrack &rNewTrack,
    int frameId,
    bool newId) {
  auto xyah = this->TlwhToXyah(rNewTrack.mTlwh);
  DetectBox xyah_box;
  xyah_box[0] = xyah[0];
  xyah_box[1] = xyah[1];
  xyah_box[2] = xyah[2];
  xyah_box[3] = xyah[3];
  auto mc = mpKalmanFilter->Update(mMean, mCovariance, xyah_box);
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
    const STrack &rNewTrack,
    int frameId,
    bool updateFeature) {
  mFrameId = frameId;
  ++mTrackletLen;

  auto xyah = this->TlwhToXyah(rNewTrack.mTlwh);
  DetectBox xyah_box;
  xyah_box[0] = xyah[0];
  xyah_box[1] = xyah[1];
  xyah_box[2] = xyah[2];
  xyah_box[3] = xyah[3];

  auto mc = mpKalmanFilter->Update(mMean, mCovariance, xyah_box);
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

void STrack::UpdateFeatures(std::vector<float> feat) {
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

