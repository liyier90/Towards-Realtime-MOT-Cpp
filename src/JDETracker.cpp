#include "JDETracker.h"

#include <fstream>
#include <iostream>
#include <unordered_map>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "Matching.h"
#include "Utils.h"

JDETracker::JDETracker(
    const std::string &rModelPath,
    int frameRate,
    int trackBuffer)
  : mNetWidth {576},
    mNetHeight {320},
    mScoreThreshold {0.5},
    mNmsThreshold {0.4},
    mMaxTimeLost {static_cast<int>(frameRate / 30.0 * trackBuffer)},
    mFrameId {0}
{
  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Test on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Test on CPU." << std::endl;
    device_type = torch::kCPU;
  }

  std::cout << "num threads: " << torch::get_num_threads() << std::endl;

  mpDevice = new torch::Device(device_type);

  std::cout << "Load model ...";
  mModel = torch::jit::load(rModelPath);
  mModel.to(*mpDevice);
  std::cout << "Done!" << std::endl;
}

JDETracker::~JDETracker() {
  delete mpDevice;
}

std::vector<STrack> JDETracker::Update(
    const cv::Mat &rPaddedImage,
    const cv::Mat &rImage) {
  auto img_tensor = torch::from_blob(rPaddedImage.data,
      {mNetHeight, mNetWidth, 3}, torch::kFloat32);
  img_tensor = torch::unsqueeze(img_tensor, 0).permute({0, 3, 1, 2});

  // Create a std::vector of inputs.
  std::vector<torch::jit::IValue> inputs = {img_tensor.to(*mpDevice)};

  /// Step 1: Network forward, get detections & embeddings
  ++mFrameId;

  // Store active tracks for the current frame
  std::vector<STrack> activated_stracks;
  // Lost tracks whose detections are obtained in the current frame
  std::vector<STrack> refind_stracks;
  // Tracks which are not obtained in the current frame but are not removed,
  // i.e., lost for some time less than threshold
  std::vector<STrack> lost_stracks;
  std::vector<STrack> removed_stracks;

  std::vector<STrack> detections;
  std::vector<STrack> detections_tmp;

  std::vector<STrack> tracked_stracks_swap;
  std::vector<STrack> res_1;
  std::vector<STrack> res_2;
  std::vector<STrack> output_stracks;

  std::vector<STrack*> unconfirmed;
  std::vector<STrack*> tracked_stracks;
  std::vector<STrack*> strack_pool;
  std::vector<STrack*> r_tracked_stracks;

  const auto pred_raw = mModel.forward(inputs)
      .toTensor()
      .to(torch::kCPU)
      .squeeze(0);
  // Filter by confidence score
  const auto pred = pred_raw.index_select(0,
      torch::nonzero(pred_raw.select(1, 4) > mScoreThreshold).squeeze());

  if (pred.sizes()[0] > 0) {
    const auto dets = jde_util::NonMaxSuppression(pred, mNmsThreshold)
        .contiguous();
    auto coords = dets.slice(1, 0, 4);
    jde_util::ScaleCoords(cv::Size(mNetWidth, mNetHeight), rImage.size(),
        &coords);
    // [x1, y1, x2, y2, object_conf, class_score, class_pred]
    for (int i = 0; i < dets.sizes()[0]; ++i) {
      std::vector<float> tlbr(dets[i].data_ptr<float>(),
          dets[i].data_ptr<float>() + 4);
      auto score = dets[i][4].item<float>();
      std::vector<float> features(dets[i].data_ptr<float>() + 6,
          dets[i].data_ptr<float>() + dets.sizes()[1]);

      STrack strack(STrack::pTlbrToTlwh(&tlbr), score, features);
      detections.push_back(strack);
    }
  }

  // Add newly detected tracklets to tracked_stracks
  for (int i = 0; i < mTrackedStracks.size(); ++i) {
    if (mTrackedStracks[i].mIsActivated) {
      tracked_stracks.push_back(&mTrackedStracks[i]);
    } else {
      unconfirmed.push_back(&mTrackedStracks[i]);
    }
  }

  /// Step 2: First association, with embedding
  strack_pool = strack_util::CombineStracks(tracked_stracks, mLostStracks);
  STrack::MultiPredict(strack_pool, mKalmanFilter);

  std::vector<std::vector<float>> dists;
  auto num_rows = 0;
  auto num_cols = 0;
  matching::EmbeddingDistance(strack_pool, detections, &dists, &num_rows,
      &num_cols);
  matching::FuseMotion(mKalmanFilter, strack_pool, detections, &dists);

  std::vector<std::vector<int>> matches;
  std::vector<int> u_track;
  std::vector<int> u_detection;
  matching::LinearAssignment(dists, num_rows, num_cols, /*threshold=*/0.7,
      &matches, &u_track, &u_detection);

  for (int i = 0; i < matches.size(); ++i) {
    STrack *p_track = strack_pool[matches[i][0]];
    STrack *p_det = &detections[matches[i][1]];
    if (p_track->mState == TrackState::Tracked) {
      p_track->Update(*p_det, mFrameId);
      activated_stracks.push_back(*p_track);
    } else {
      p_track->ReActivate(*p_det, mFrameId);
      refind_stracks.push_back(*p_track);
    }
  }

  /// Step 3: Second association, with IOU
  for (int i = 0; i < u_detection.size(); ++i) {
    detections_tmp.push_back(detections[u_detection[i]]);
  }
  detections = detections_tmp;

  for (int i = 0; i < u_track.size(); ++i) {
    if (strack_pool[u_track[i]]->mState == TrackState::Tracked) {
      r_tracked_stracks.push_back(strack_pool[u_track[i]]);
    }
  }

  dists = matching::IouDistance(r_tracked_stracks, detections, &num_rows,
      &num_cols);

  matches.clear();
  u_track.clear();
  u_detection.clear();
  matching::LinearAssignment(dists, num_rows, num_cols, /*threshold=*/0.7,
      &matches, &u_track, &u_detection);

  for (int i = 0; i < matches.size(); ++i) {
    auto *p_track = r_tracked_stracks[matches[i][0]];
    auto *p_det = &detections[matches[i][1]];
    if (p_track->mState == TrackState::Tracked) {
      p_track->Update(*p_det, mFrameId);
      activated_stracks.push_back(*p_track);
    } else {
      p_track->ReActivate(*p_det, mFrameId);
      refind_stracks.push_back(*p_track);
    }
  }

  for (int i = 0; i < u_track.size(); ++i) {
    auto *p_track = r_tracked_stracks[u_track[i]];
    if (p_track->mState != TrackState::Lost) {
      p_track->MarkLost();
      lost_stracks.push_back(*p_track);
    }
  }

  // Deal with unconfirmed tracks, usually tracks with only one beginning frame
  detections_tmp.clear();
  for (int i = 0; i < u_detection.size(); ++i) {
    detections_tmp.push_back(detections[u_detection[i]]);
  }
  detections = detections_tmp;

  dists = matching::IouDistance(unconfirmed, detections, &num_rows, &num_cols);

  matches.clear();
  std::vector<int> u_unconfirmed;
  u_detection.clear();
  matching::LinearAssignment(dists, num_rows, num_cols, /*threshold=*/0.7,
      &matches, &u_unconfirmed, &u_detection);

  for (int i = 0; i < matches.size(); ++i) {
    unconfirmed[matches[i][0]]->Update(detections[matches[i][1]], mFrameId);
    activated_stracks.push_back(*unconfirmed[matches[i][0]]);
  }

  for (int i = 0; i < u_unconfirmed.size(); ++i) {
    auto *p_track = unconfirmed[u_unconfirmed[i]];
    p_track->MarkRemoved();
    removed_stracks.push_back(*p_track);
  }

  /// Step 4: Init new stracks
  for (int i = 0; i < u_detection.size(); ++i) {
    auto *p_track = &detections[u_detection[i]];
    if (p_track->mScore < mScoreThreshold) {
      continue;
    }
    p_track->Activate(&mKalmanFilter, mFrameId);
    activated_stracks.push_back(*p_track);
  }

  /// Step 5: Update state
  for (int i = 0; i < mLostStracks.size(); ++i) {
    if (mFrameId - mLostStracks[i].EndFrame() > mMaxTimeLost) {
      mLostStracks[i].MarkRemoved();
      removed_stracks.push_back(mLostStracks[i]);
    }
  }

  for (int i = 0; i < mTrackedStracks.size(); ++i) {
    if (mTrackedStracks[i].mState == TrackState::Tracked) {
      tracked_stracks_swap.push_back(mTrackedStracks[i]);
    }
  }
  mTrackedStracks = tracked_stracks_swap;

  mTrackedStracks = strack_util::CombineStracks(mTrackedStracks,
      activated_stracks);
  mTrackedStracks = strack_util::CombineStracks(mTrackedStracks,
      refind_stracks);

  mLostStracks = strack_util::SubstractStracks(mLostStracks, mTrackedStracks);
  for (int i = 0; i < lost_stracks.size(); ++i) {
    mLostStracks.push_back(lost_stracks[i]);
  }

  mLostStracks = strack_util::SubstractStracks(mLostStracks, mRemovedStracks);
  for (int i = 0; i < removed_stracks.size(); ++i) {
    mRemovedStracks.push_back(removed_stracks[i]);
  }

  strack_util::RemoveDuplicateStracks(mTrackedStracks, mLostStracks, &res_1,
      &res_2);

  mTrackedStracks = res_1;
  mLostStracks = res_2;

  for (int i = 0; i < mTrackedStracks.size(); ++i) {
    if (mTrackedStracks[i].mIsActivated) {
      output_stracks.push_back(mTrackedStracks[i]);
    }
  }
  return output_stracks;
}

cv::Mat JDETracker::Preprocess(cv::Mat image) {
  auto padded_image = jde_util::Letterbox(image, mNetHeight, mNetWidth);

  cv::cvtColor(padded_image, padded_image, cv::COLOR_BGR2RGB);
  padded_image.convertTo(padded_image, CV_32FC3);
  padded_image /= 255.0;

  return padded_image;
}

namespace strack_util {
std::vector<STrack*> CombineStracks(
    const std::vector<STrack*> &rStracks1,
    const std::vector<STrack> &rStracks2) {
  std::unordered_map<int, int> exists;
  std::vector<STrack*> res;
  for (int i = 0; i < rStracks1.size(); ++i) {
    exists.insert(std::pair<int, int>(rStracks1[i]->mTrackId, 1));
    res.push_back(rStracks1[i]);
  }
  for (int i = 0; i < rStracks2.size(); ++i) {
    auto tid = rStracks2[i].mTrackId;
    if (!exists[tid] || exists.count(tid) == 0) {
      exists[tid] = 1;
      res.push_back(const_cast<STrack*>(&rStracks2[i]));
    }
  }
  return res;
}

std::vector<STrack> CombineStracks(
    const std::vector<STrack> &rStracks1,
    const std::vector<STrack> &rStracks2) {
  std::unordered_map<int, int> exists;
  std::vector<STrack> res;
  for (int i = 0; i < rStracks1.size(); ++i) {
    exists.insert(std::pair<int, int>(rStracks1[i].mTrackId, 1));
    res.push_back(rStracks1[i]);
  }
  for (int i = 0; i < rStracks2.size(); ++i) {
    auto tid = rStracks2[i].mTrackId;
    if (!exists[tid] || exists.count(tid) == 0) {
      exists[tid] = 1;
      res.push_back(rStracks2[i]);
    }
  }
  return res;
}

void RemoveDuplicateStracks(
    const std::vector<STrack> &rStracks1,
    const std::vector<STrack> &rStracks2,
    std::vector<STrack> *pRes1,
    std::vector<STrack> *pRes2) {
  auto distances = matching::IouDistance(rStracks1, rStracks2);
  std::vector<std::pair<int, int>> pairs;
  for (int i = 0; i < distances.size(); ++i) {
    for (int j = 0; j < distances[i].size(); ++j) {
      if (distances[i][j] < 0.15) {
        pairs.push_back(std::pair<int, int>(i, j));
      }
    }
  }

  std::vector<int> duplicates_1;
  std::vector<int> duplicates_2;
  for (int i = 0; i < pairs.size(); ++i) {
    auto idx_1 = pairs[i].first;
    auto idx_2 = pairs[i].second;
    auto age_1 = rStracks1[idx_1].mFrameId - rStracks1[idx_1].mStartFrame;
    auto age_2 = rStracks2[idx_2].mFrameId - rStracks2[idx_2].mStartFrame;
    if (age_1 > age_2) {
      duplicates_2.push_back(idx_2);
    } else {
      duplicates_1.push_back(idx_1);
    }
  }

  for (int i = 0; i < rStracks1.size(); ++i) {
    auto iter = std::find(duplicates_1.begin(), duplicates_1.end(), i);
    if (iter == duplicates_1.end()) {
      pRes1->push_back(rStracks1[i]);
    }
  }

  for (int i = 0; i < rStracks2.size(); ++i) {
    auto iter = std::find(duplicates_2.begin(), duplicates_2.end(), i);
    if (iter == duplicates_2.end()) {
      pRes2->push_back(rStracks2[i]);
    }
  }
}

std::vector<STrack> SubstractStracks(
    const std::vector<STrack> &rStracks1,
    const std::vector<STrack> &rStracks2) {
  std::unordered_map<int, STrack> stracks;
  for (int i = 0; i < rStracks1.size(); ++i) {
    stracks.insert(
        std::pair<int, STrack>(rStracks1[i].mTrackId, rStracks1[i]));
  }
  for (int i = 0; i < rStracks2.size(); ++i) {
    auto tid = rStracks2[i].mTrackId;
    if (stracks.count(tid) != 0) {
      stracks.erase(tid);
    }
  }
  std::vector<STrack> res;
  for (auto it = stracks.begin(); it != stracks.end(); ++it) {
    res.push_back(it->second);
  }

  return res;
}
}  // namespace strack_util
