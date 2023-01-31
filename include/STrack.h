#ifndef INCLUDE_STRACK_H_
#define INCLUDE_STRACK_H_

#include <vector>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>  // NOLINT
#include <torch/script.h>  // NOLINT

#include "KalmanFilter.h"

enum TrackState { New = 0, Tracked, Lost, Removed };

class STrack {
 public:
  STrack(
      std::vector<float> *pTlwh,
      float score,
      std::vector<float> features,
      int bufferSize = 30);
  ~STrack();

  static std::vector<float>* pTlbrToTlwh(std::vector<float> *pTlbr);

  static void MultiPredict(
      const std::vector<STrack*> &rStracks,
      const jde_kalman::KalmanFilter &rKalmanFilter);

  void StaticTlwh();
  void StaticTlbr();

  std::vector<float> TlwhToXyah(const std::vector<float> &rTlwh) const;
  std::vector<float> ToXyah() const;

  void MarkLost();
  void MarkRemoved();
  int NextId();
  int EndFrame();

  void Activate(
      const jde_kalman::KalmanFilter &rKalmanFilter,
      int frameId);

  void ReActivate(
      const STrack &rNewTrack,
      int frameId,
      bool newId = false);

  void Update(
      const STrack &rNewTrack,
      int frameId,
      bool updateFeature = true);

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

  KalmanMean mMean;
  KalmanCov mCovariance;

 private:
  void UpdateFeatures(std::vector<float> feat);
  jde_kalman::KalmanFilter mKalmanFilter;
};

#endif  // INCLUDE_STRACK_H_

