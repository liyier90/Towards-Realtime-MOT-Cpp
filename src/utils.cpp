/*
author: samylee
github: https://github.com/samylee
date: 08/19/2021
*/

#include "JDETracker.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "lapjv.h"

cv::Size JDETracker::GetSize(
    int vw
  , int vh
  , int dw
  , int dh)
{
  cv::Size size;
  float wa = static_cast<float>(dw) / vw;
  float ha = static_cast<float>(dh) / vh;
  float a = min(wa, ha);
  size.width = static_cast<int>(vw * a);
  size.height = static_cast<int>(vh * a);

  return size;
}

cv::Mat JDETracker::Letterbox(
    cv::Mat img
  , int height
  , int width)
{
  cv::Size shape = cv::Size(img.cols, img.rows);
  auto ratio = min(static_cast<float>(height) / shape.height,
      static_cast<float>(width) / shape.width);
  auto new_shape = cv::Size(round(shape.width * ratio),
      round(shape.height * ratio));
  auto dw = static_cast<float>(width - new_shape.width) / 2;
  auto dh = static_cast<float>(height - new_shape.height) / 2;
  int top = round(dh - 0.1);
  int bottom = round(dh + 0.1);
  int left = round(dw - 0.1);
  int right = round(dw + 0.1);

  cv::resize(img, img, new_shape, cv::INTER_AREA);
  cv::copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT,
      cv::Scalar(127.5, 127.5, 127.5));
  return img;
}

torch::Tensor JDETracker::Nms(
    const torch::Tensor &rBoxes
  , const torch::Tensor &rScores
  , float nmsThreshold)
{
  auto num_to_keep = 0;
  auto top_k = 200;

  auto x1 = rBoxes.select(1, 0).contiguous();
  auto y1 = rBoxes.select(1, 1).contiguous();
  auto x2 = rBoxes.select(1, 2).contiguous();
  auto y2 = rBoxes.select(1, 3).contiguous();

  torch::Tensor areas = (x2 - x1) * (y2 - y1);
  
  auto order = std::get<1>(
      rScores.sort(/*stable=*/true, /*dim=*/0, /*descending=*/true));

  auto num_dets = rScores.size(0);
  torch::Tensor keep = torch::empty({num_dets}).to(torch::kLong);
  torch::Tensor suppressed = torch::empty({num_dets}).to(torch::kByte);

  auto *p_keep = keep.data_ptr<long>();
  auto *p_suppressed = suppressed.data_ptr<uint8_t>();
  auto *p_order = order.data_ptr<long>();
  auto *p_x1 = x1.data_ptr<float>();
  auto *p_y1 = y1.data_ptr<float>();
  auto *p_x2 = x2.data_ptr<float>();
  auto *p_y2 = y2.data_ptr<float>();
  auto *p_areas = areas.data_ptr<float>();

  for (int _i = 0; _i < num_dets; ++_i) {
    if (num_to_keep >= top_k) {
      break;
    }
    auto i = p_order[_i];
    if (p_suppressed[i] == 1) {
      continue;
    }
    p_keep[num_to_keep++] = i;
    auto i_x1 = p_x1[i];
    auto i_y1 = p_y1[i];
    auto i_x2 = p_x2[i];
    auto i_y2 = p_y2[i];
    auto i_area = p_areas[i];

    for (int _j = _i + 1; _j < num_dets; ++_j) {
      auto j = p_order[_j];
      if (p_suppressed[j] == 1) {
        continue;
      }
      auto xx1 = std::max(i_x1, p_x1[j]);
      auto yy1 = std::max(i_y1, p_y1[j]);
      auto xx2 = std::min(i_x2, p_x2[j]);
      auto yy2 = std::min(i_y2, p_y2[j]);
      auto w = std::max(static_cast<float>(0), xx2 - xx1);
      auto h = std::max(static_cast<float>(0), yy2 - yy1);
      auto inter = w * h;
      auto iou = inter / (i_area + p_areas[j] - inter);
      if (iou > nmsThreshold) {
        p_suppressed[j] = 1;
      }
    }
  }
  return keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep);
}

torch::Tensor JDETracker::xywh2xyxy(torch::Tensor x)
{
  auto y = torch::zeros_like(x);
  y.slice(1, 0, 1) = x.slice(1, 0, 1) - x.slice(1, 2, 3) / 2;
  y.slice(1, 1, 2) = x.slice(1, 1, 2) - x.slice(1, 3, 4) / 2;
  y.slice(1, 2, 3) = x.slice(1, 0, 1) + x.slice(1, 2, 3) / 2;
  y.slice(1, 3, 4) = x.slice(1, 1, 2) + x.slice(1, 3, 4) / 2;

  return y;
}

torch::Tensor JDETracker::NonMaxSuppression(torch::Tensor prediction)
{
  prediction.slice(1, 0, 4) = xywh2xyxy(prediction.slice(1, 0, 4));
  torch::Tensor nms_indices = this->Nms(prediction.slice(1, 0, 4),
      prediction.select(1, 4), mNmsThreshold);

  return prediction.index_select(0, nms_indices);
}

void JDETracker::ScaleCoords(
    torch::Tensor &coords
  , Size img_size
  , Size img0_shape)
{
  float gain_w = static_cast<float>(img_size.width) / img0_shape.width;
  float gain_h = static_cast<float>(img_size.height) / img0_shape.height;
  float gain = min(gain_w, gain_h);
  float pad_x = (img_size.width - img0_shape.width*gain) / 2;
  float pad_y = (img_size.height - img0_shape.height*gain) / 2;
  coords.select(1, 0) -= pad_x;
  coords.select(1, 1) -= pad_y;
  coords.select(1, 2) -= pad_x;
  coords.select(1, 3) -= pad_y;
  coords /= gain;
  coords = torch::clamp(coords, 0);
}

std::vector<STrack*> JDETracker::CombineStracks(
    std::vector<STrack*> &rStracks1
  , std::vector<STrack> &rStracks2)
{
  std::map<int, int> exists;
  std::vector<STrack*> res;
  for (int i = 0; i < rStracks1.size(); ++i) {
    exists.insert(std::pair<int, int>(rStracks1[i]->mTrackId, 1));
    res.push_back(rStracks1[i]);
  }
  for (int i = 0; i < rStracks2.size(); ++i) {
    auto tid = rStracks2[i].mTrackId;
    if (!exists[tid] || exists.count(tid) == 0) {
      exists[tid] = 1;
      res.push_back(&rStracks2[i]);
    }
  }
  return res;
}

std::vector<STrack> JDETracker::CombineStracks(
    std::vector<STrack>& rStracks1 
  , std::vector<STrack>& rStracks2)
{
  std::map<int, int> exists;
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

std::vector<STrack> JDETracker::SubstractStracks(
    std::vector<STrack> &rStracks1
  , std::vector<STrack> &rStracks2)
{
  std::map<int, STrack> stracks;
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

void JDETracker::RemoveDuplicateStracks(
    const std::vector<STrack> &rStracks1
  , const std::vector<STrack> &rStracks2
  , std::vector<STrack> &rRes1
  , std::vector<STrack> &rRes2)
{
  auto distances = this->IouDistance(rStracks1, rStracks2);
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
      rRes1.push_back(rStracks1[i]);
    }
  }

  for (int i = 0; i < rStracks2.size(); ++i) {
    auto iter = std::find(duplicates_2.begin(), duplicates_2.end(), i);
    if (iter == duplicates_2.end()) {
      rRes2.push_back(rStracks2[i]);
    }
  }
}

void JDETracker::EmbeddingDistance(
    std::vector<STrack*> &rTracks
  , std::vector<STrack> &rDetections
  , std::vector<std::vector<float>> &rCostMatrix
  , int *pNumRows
  , int *pNumCols)
{
  if (rTracks.size() * rDetections.size() == 0) {
    *pNumRows = rTracks.size();
    *pNumCols = rDetections.size();
    return;
  }

  for (int i = 0; i < rTracks.size(); ++i) {
    std::vector<float> cost_matrix_row;
    auto track_feature = rTracks[i]->mSmoothFeat;
    for (int j = 0; j < rDetections.size(); ++j) {
      auto det_feature = rDetections[j].mCurrFeat;
      float feat_square = 0.0;
      for (int k = 0; k < det_feature.size(); ++k) {
        feat_square += (track_feature[k] - det_feature[k]) *
            (track_feature[k] - det_feature[k]);
      }
      cost_matrix_row.push_back(std::sqrt(feat_square));
    }
    rCostMatrix.push_back(cost_matrix_row);
  }
  *pNumRows = rTracks.size();
  *pNumCols = rDetections.size();
}

void JDETracker::FuseMotion(
    std::vector<std::vector<float>> &rCostMatrix
  , std::vector<STrack*> &rTracks
  , std::vector<STrack> &rDetections
  , bool onlyPosition
  , float coeff)
{
  if (rCostMatrix.size() == 0) {
    return;
  }

  auto gating_dim = onlyPosition ? 2 : 4;
  float gating_threshold = mKalmanFilter.chi2inv95[gating_dim];

  std::vector<DETECTBOX> measurements;
  for (int i = 0; i < rDetections.size(); ++i) {
    DETECTBOX measurement;
    std::vector<float> tlwh = rDetections[i].to_xyah();
    measurement[0] = tlwh[0];
    measurement[1] = tlwh[1];
    measurement[2] = tlwh[2];
    measurement[3] = tlwh[3];
    measurements.push_back(measurement);
  }

  for (int i = 0; i < rTracks.size(); ++i) {
    auto gating_distance = mKalmanFilter.gating_distance(rTracks[i]->mMean,
        rTracks[i]->mCovariance, measurements, onlyPosition);

    for (int j = 0; j < rCostMatrix[i].size(); ++j) {
      if (gating_distance[j] > gating_threshold) {
        rCostMatrix[i][j] = FLT_MAX;
      }
      rCostMatrix[i][j] = coeff * rCostMatrix[i][j] + (1 - coeff) *
          gating_distance[j];
    }
  }
}

void JDETracker::LinearAssignment(
    std::vector<std::vector<float>> &rCostMatrix
  , int numRows
  , int numCols
  , float threshold
  , std::vector<std::vector<int>> &rMatches
  , std::vector<int> &rUnmatched1
  , std::vector<int> &rUnmatched2)
{
  if (rCostMatrix.size() == 0) {
    for (int i = 0; i < numRows; ++i) {
      rUnmatched1.push_back(i);
    }
    for (int i = 0; i < numCols; ++i) {
      rUnmatched2.push_back(i);
    }
    return;
  }

  std::vector<int> rowsol;
  std::vector<int> colsol;
  auto c = this->lapjv(rCostMatrix, rowsol, colsol, true, threshold);
  for (int i = 0; i < rowsol.size(); ++i) {
    if (rowsol[i] >= 0) {
      std::vector<int> match;
      match.push_back(i);
      match.push_back(rowsol[i]);
      rMatches.push_back(match);
    } else {
      rUnmatched1.push_back(i);
    }
  }

  for (int i = 0; i < colsol.size(); ++i) {
    if (colsol[i] < 0) {
      rUnmatched2.push_back(i);
    }
  }
}

std::vector<std::vector<float>> JDETracker::Ious(
    std::vector<std::vector<float>> &atlbrs
  , std::vector<std::vector<float>> &btlbrs)
{
  std::vector<std::vector<float>> ious;
  if (atlbrs.size()*btlbrs.size() == 0) {
    return ious;
  }

  ious.resize(atlbrs.size());
  for (int i = 0; i < ious.size(); i++) {
    ious[i].resize(btlbrs.size());
  }

  // bbox_ious
  for (int k = 0; k < btlbrs.size(); k++) {
    std::vector<float> ious_tmp;
    float box_area = (btlbrs[k][2] - btlbrs[k][0] + 1) *
        (btlbrs[k][3] - btlbrs[k][1] + 1);
    for (int n = 0; n < atlbrs.size(); n++) {
      float iw = std::min(atlbrs[n][2], btlbrs[k][2]) -
          std::max(atlbrs[n][0], btlbrs[k][0]) + 1;
      if (iw > 0) {
        float ih = std::min(atlbrs[n][3], btlbrs[k][3]) -
            std::max(atlbrs[n][1], btlbrs[k][1]) + 1;
        if (ih > 0) {
          float ua = (atlbrs[n][2] - atlbrs[n][0] + 1) *
              (atlbrs[n][3] - atlbrs[n][1] + 1) + box_area - iw * ih;
          ious[n][k] = iw * ih / ua;
        } else {
          ious[n][k] = 0.0;
        }
      } else {
        ious[n][k] = 0.0;
      }
    }
  }

  return ious;
}

std::vector<std::vector<float>> JDETracker::IouDistance(
    const std::vector<STrack*> &rTracks1
  , const std::vector<STrack> &rTracks2
  , int &rNumRows
  , int &rNumCols)
{
  std::vector<std::vector<float>> tlbrs_1;
  std::vector<std::vector<float>> tlbrs_2;
  for (int i = 0; i < rTracks1.size(); ++i) {
    tlbrs_1.push_back(rTracks1[i]->mTlbr);
  }
  for (int i = 0; i < rTracks2.size(); ++i) {
    tlbrs_2.push_back(rTracks2[i].mTlbr);
  }

  rNumRows = rTracks1.size();
  rNumCols = rTracks2.size();

  std::vector<std::vector<float>> ious = this->Ious(tlbrs_1, tlbrs_2);
  std::vector<std::vector<float>> cost_matrix;
  for (int i = 0; i < ious.size(); ++i) {
    std::vector<float> iou;
    for (int j = 0; j < ious[i].size(); ++j) {
      iou.push_back(1 - ious[i][j]);
    }
    cost_matrix.push_back(iou);
  }

  return cost_matrix;
}

std::vector<std::vector<float>> JDETracker::IouDistance(
    const std::vector<STrack> &rTracks1
  , const std::vector<STrack> &rTracks2)
{
  std::vector<std::vector<float>> tlbrs_1;
  std::vector<std::vector<float>> tlbrs_2;
  for (int i = 0; i < rTracks1.size(); ++i) {
    tlbrs_1.push_back(rTracks1[i].mTlbr);
  }
  for (int i = 0; i < rTracks2.size(); ++i) {
    tlbrs_2.push_back(rTracks2[i].mTlbr);
  }

  std::vector<std::vector<float>> ious = this->Ious(tlbrs_1, tlbrs_2);
  std::vector<std::vector<float>> cost_matrix;
  for (int i = 0; i < ious.size(); ++i) {
    std::vector<float> iou;
    for (int j = 0; j < ious[i].size(); ++j) {
      iou.push_back(1 - ious[i][j]);
    }
    cost_matrix.push_back(iou);
  }

  return cost_matrix;
}

double JDETracker::lapjv(
    const std::vector<std::vector<float>> &cost
  , std::vector<int>& rowsol
  , std::vector<int>& colsol
  , bool extend_cost
  , float cost_limit
  , bool return_cost)
{
  std::vector<std::vector<float> > cost_c;
  cost_c.assign(cost.begin(), cost.end());

  std::vector<std::vector<float> > cost_c_extended;

  int n_rows = cost.size();
  int n_cols = cost[0].size();
  rowsol.resize(n_rows);
  colsol.resize(n_cols);

  int n = 0;
  if (n_rows == n_cols) {
    n = n_rows;
  } else if (!extend_cost) {
    std::cout << "set extend_cost=True" << std::endl;
    system("pause");
    exit(0);
  }

  if (extend_cost || cost_limit < FLT_MAX) {
    n = n_rows + n_cols;
    cost_c_extended.resize(n);
    for (int i = 0; i < cost_c_extended.size(); i++) {
      cost_c_extended[i].resize(n);
    }

    if (cost_limit < FLT_MAX) {
      for (int i = 0; i < cost_c_extended.size(); i++) {
        for (int j = 0; j < cost_c_extended[i].size(); j++) {
          cost_c_extended[i][j] = cost_limit / 2.0;
        }
      }
    } else {
      float cost_max = -1;
      for (int i = 0; i < cost_c.size(); i++) {
        for (int j = 0; j < cost_c[i].size(); j++) {
          if (cost_c[i][j] > cost_max) {
            cost_max = cost_c[i][j];
          }
        }
      }
      for (int i = 0; i < cost_c_extended.size(); i++) {
        for (int j = 0; j < cost_c_extended[i].size(); j++) {
          cost_c_extended[i][j] = cost_max + 1;
        }
      }
    }

    for (int i = n_rows; i < cost_c_extended.size(); i++) {
      for (int j = n_cols; j < cost_c_extended[i].size(); j++) {
        cost_c_extended[i][j] = 0;
      }
    }
    for (int i = 0; i < n_rows; i++) {
      for (int j = 0; j < n_cols; j++) {
        cost_c_extended[i][j] = cost_c[i][j];
      }
    }

    cost_c.clear();
    cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
  }

  double** cost_ptr;
  cost_ptr = new double *[sizeof(double *) * n];
  for (int i = 0; i < n; i++) {
    cost_ptr[i] = new double[sizeof(double) * n];
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      cost_ptr[i][j] = cost_c[i][j];
    }
  }

  int* x_c = new int[sizeof(int) * n];
  int* y_c = new int[sizeof(int) * n];

  int ret = lapjv_internal(n, cost_ptr, x_c, y_c);
  if (ret != 0) {
    std::cout << "Calculate Wrong!" << std::endl;
    system("pause");
    exit(0);
  }

  double opt = 0.0;

  if (n != n_rows) {
    for (int i = 0; i < n; i++) {
      if (x_c[i] >= n_cols) {
        x_c[i] = -1;
      }
      if (y_c[i] >= n_rows) {
        y_c[i] = -1;
      }
    }
    for (int i = 0; i < n_rows; i++) {
      rowsol[i] = x_c[i];
    }
    for (int i = 0; i < n_cols; i++) {
      colsol[i] = y_c[i];
    }

    if (return_cost) {
      for (int i = 0; i < rowsol.size(); i++) {
        if (rowsol[i] != -1) {
          // cout << i << "\t" << rowsol[i] << "\t"
          //     << cost_ptr[i][rowsol[i]] << endl;
          opt += cost_ptr[i][rowsol[i]];
        }
      }
    }
  } else if (return_cost) {
    for (int i = 0; i < rowsol.size(); i++) {
      opt += cost_ptr[i][rowsol[i]];
    }
  }

  for (int i = 0; i < n; i++) {
    delete[]cost_ptr[i];
  }
  delete[]cost_ptr;
  delete[]x_c;
  delete[]y_c;

  return opt;
}

Scalar JDETracker::get_color(int idx)
{
  idx += 3;
  return Scalar(37 * idx % 255, 17 * idx % 255, 29 * idx % 255);
}
