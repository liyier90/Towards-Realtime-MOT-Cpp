/*
author: samylee
github: https://github.com/samylee
date: 08/19/2021
*/

#include "JDETracker.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

JDETracker::JDETracker(
    const std::string &rModelPath
  , int frameRate
  , int trackBuffer)
  : mNetWidth {576},
    mNetHeight {320},
    mScoreThreshold {0.5},
    mNmsThreshold {0.4},
    mFrameId {0},
    mMaxTimeLost {static_cast<int>(frameRate / 30.0 * trackBuffer)}
{
  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Test on GPU." << std::endl;
    device_type = torch::kCUDA;
  }
  else {
    std::cout << "Test on CPU." << std::endl;
    device_type = torch::kCPU;
  }

  torch::set_num_threads(1);
  std::cout << "set threads num: " << torch::get_num_threads() << std::endl;

  mpDevice = new torch::Device(device_type);

  std::cout << "Load model ... ";
  jde_model = torch::jit::load(rModelPath);
  jde_model.to(*mpDevice);
  std::cout << "Done!" << std::endl;
}

JDETracker::~JDETracker()
{
  delete mpDevice;
}

void JDETracker::Update(const std::string &rVideoPath)
{
  cv::VideoCapture cap(rVideoPath);
  if (!cap.isOpened()) {
    return;
  }

  int vw = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int vh = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  auto size = this->GetSize(vw, vh, mNetWidth, mNetHeight);

  cv::Mat img0;
  while (true) {
    cap >> img0;
    if (img0.empty()) {
      break;
    }

    cv::resize(img0, img0, size);
    auto img = this->Letterbox(img0, mNetHeight, mNetWidth);

    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    cv::Mat img_float;
    img_rgb.convertTo(img_float, CV_32FC3);
    img_float /= 255.0;

    auto img_tensor = torch::from_blob(img_float.data,
        {mNetHeight, mNetWidth, 3}, torch::kFloat32);
    auto img_tensor_unsqueeze = torch::unsqueeze(img_tensor, 0)
        .permute({0, 3, 1, 2});
    // Create a std::vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(img_tensor_unsqueeze.to(*mpDevice));

    /// Step 1: Network forward, get detections & embeddings
    ++mFrameId;
    std::cout << mFrameId << " mTrackedStracks " << mTrackedStracks.size() << std::endl;
    for (int i = 0; i < mTrackedStracks.size(); ++i) {
      std::cout << mTrackedStracks[i].mTrackId << " " << mTrackedStracks[i].mState << ", ";
    }
    std::cout << std::endl;

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

    auto pred_raw = jde_model.forward(inputs)
        .toTensor()
        .to(torch::kCPU)
        .squeeze(0);
    auto pred = pred_raw.index_select(0,
        torch::nonzero(pred_raw.select(1, 4) > mScoreThreshold).squeeze());

    if (pred.sizes()[0] > 0) {
      auto dets = this->NonMaxSuppression(pred);
      auto coords = dets.slice(1, 0, 4);
      this->ScaleCoords(coords, cv::Size(mNetWidth, mNetHeight),
          cv::Size(img0.cols, img0.rows));
      // [x1, y1, x2, y2, object_conf, class_score, class_pred]
      for (int i = 0; i < dets.sizes()[0]; ++i) {
        std::vector<float> tlbr(4);
        tlbr[0] = dets[i][0].item<float>();
        tlbr[1] = dets[i][1].item<float>();
        tlbr[2] = dets[i][2].item<float>();
        tlbr[3] = dets[i][3].item<float>();

        cv::rectangle(img0, cv::Rect(cv::Point(tlbr[0], tlbr[1]),
              cv::Point(tlbr[2], tlbr[3])), cv::Scalar(0, 255, 0), 2);

        auto score = dets[i][4].item<float>();

        std::vector<float> features;
        for (int j = 6; j < dets.sizes()[1]; ++j) {
          features.push_back(dets[i][j].item<float>());
        }

        STrack strack(STrack::TlbrToTlwh(tlbr), score, features);
        detections.push_back(strack);
      }
    }

    // Add newly detected tracklets to tracked_stracks
    std::cout << "before tracked_stracks" << std::endl;
    std::cout << "mTrackedStracks " << mTrackedStracks.size() << std::endl;
    for (int i = 0; i < mTrackedStracks.size(); ++i) {
      std::cout << mTrackedStracks[i].mTrackId << " " << mTrackedStracks[i].mState << ", ";
    }
    std::cout << std::endl;
    for (int i = 0; i < mTrackedStracks.size(); ++i) {
      if (mTrackedStracks[i].mIsActivated) {
        tracked_stracks.push_back(&mTrackedStracks[i]);
      } else {
        unconfirmed.push_back(&mTrackedStracks[i]);
      }
    }
    std::cout << tracked_stracks.size() << std::endl;

    /// Step 2: First association, with embedding
    strack_pool = this->CombineStracks(tracked_stracks, mLostStracks);
    STrack::MultiPredict(strack_pool, mKalmanFilter);

    std::vector<std::vector<float>> dists;
    auto num_rows = 0;
    auto num_cols = 0;
    std::cout << "strack_pool " << strack_pool.size()
        << " detections " << detections.size() << std::endl;
    this->EmbeddingDistance(strack_pool, detections, dists, &num_rows,
        &num_cols);
    for (int i = 0; i < num_rows; ++i) {
      for (int j = 0; j < num_cols; ++j) {
        std::cout << dists[i][j] << " ";
      }
      std::cout << std::endl;
    }
    this->FuseMotion(dists, strack_pool, detections);

    std::vector<std::vector<int>> matches;
    std::vector<int> u_track;
    std::vector<int> u_detection;
    this->LinearAssignment(dists, num_rows, num_cols, 0.7, matches,
        u_track, u_detection);

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
    // for (int i = 0; i < u_detection.size(); ++i) {
    //   detections_tmp.push_back(detections[u_detection[i]]);
    // }
    // detections.clear();
    // detections.assign(detections_tmp.begin(), detections_tmp.end());
    //
    // for (int i = 0; i < u_track.size(); i++)
    // {
    //   if (strack_pool[u_track[i]]->mState == TrackState::Tracked)
    //   {
    //     r_tracked_stracks.push_back(strack_pool[u_track[i]]);
    //   }
    // }
    //
    // dists.clear();
    // dists = iou_distance(r_tracked_stracks, detections, dist_size, dist_size_size);
    //
    // matches.clear();
    // u_track.clear();
    // u_detection.clear();
    // LinearAssignment(dists, dist_size, dist_size_size, 0.7, matches, u_track, u_detection);
    //
    // for (int i = 0; i < matches.size(); i++) {
    //   STrack *track = r_tracked_stracks[matches[i][0]];
    //   STrack *det = &detections[matches[i][1]];
    //   if (track->mState == TrackState::Tracked) {
    //     track->Update(*det, this->mFrameId);
    //     activated_stracks.push_back(*track);
    //   } else {
    //     track->ReActivate(*det, this->mFrameId, false);
    //     refind_stracks.push_back(*track);
    //   }
    // }
    //
    // for (int i = 0; i < u_track.size(); i++) {
    //   STrack *track = r_tracked_stracks[u_track[i]];
    //   if (track->mState != TrackState::Lost) {
    //     track->mark_lost();
    //     lost_stracks.push_back(*track);
    //   }
    // }

    // Deal with unconfirmed tracks, usually tracks with only one beginning frame
    detections_tmp.clear();
    for (int i = 0; i < u_detection.size(); ++i) {
      detections_tmp.push_back(detections[u_detection[i]]);
    }
    detections.clear();
    detections.assign(detections_tmp.begin(), detections_tmp.end());

    dists.clear();
    dists = this->IouDistance(unconfirmed, detections, num_rows, num_cols);

    matches.clear();
    std::vector<int> u_unconfirmed;
    u_detection.clear();
    this->LinearAssignment(dists, num_rows, num_cols, /*threshold=*/0.7,
        matches, u_unconfirmed, u_detection);

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
      p_track->Activate(mKalmanFilter, mFrameId);
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
    mTrackedStracks.clear();
    mTrackedStracks.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

    mTrackedStracks = this->CombineStracks(mTrackedStracks, activated_stracks);
    mTrackedStracks = this->CombineStracks(mTrackedStracks, refind_stracks);

    mLostStracks = this->SubstractStracks(mLostStracks, mTrackedStracks);
    for (int i = 0; i < lost_stracks.size(); ++i) {
      mLostStracks.push_back(lost_stracks[i]);
    }

    mLostStracks = this->SubstractStracks(mLostStracks, mRemovedStracks);
    for (int i = 0; i < removed_stracks.size(); ++i) {
      mRemovedStracks.push_back(removed_stracks[i]);
    }

    this->RemoveDuplicateStracks(mTrackedStracks, mLostStracks, res_1, res_2);

    mTrackedStracks.clear();
    mTrackedStracks.assign(res_1.begin(), res_1.end());
    mLostStracks.clear();
    mLostStracks.assign(res_2.begin(), res_2.end());

    for (int i = 0; i < mTrackedStracks.size(); ++i) {
      if (mTrackedStracks[i].mIsActivated) {
        output_stracks.push_back(mTrackedStracks[i]);
      }
    }

    for (int i = 0; i < output_stracks.size(); ++i) {
      std::vector<float> tlwh = output_stracks[i].mTlwh;
      bool vertical = tlwh[2] / tlwh[3] > 1.6;
      if (tlwh[2] * tlwh[3] > 200 && !vertical) {
        Scalar s = get_color(output_stracks[i].mTrackId);
        rectangle(img0, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
        // putText(img0, to_string(output_stracks[i].mTrackId),
        //     Point(tlwh[0], tlwh[1]), 0, 0.6, s, 2);
      }
    }

    cv::imshow("test", img0);
    if (waitKey(1) > 0) {
      break;
    }
  }
  cap.release();
}
