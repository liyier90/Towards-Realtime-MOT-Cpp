#include "Matching.h"

#include "DataType.h"
#include "KalmanFilter.h"
#include "Utils.h"

namespace matching {
void EmbeddingDistance(
    const std::vector<STrack*> &rTracks,
    const std::vector<STrack> &rDetections,
    std::vector<std::vector<float>> *pCostMatrix,
    int *pNumRows,
    int *pNumCols) {
  if (rTracks.size() * rDetections.size() == 0) {
    *pNumRows = rTracks.size();
    *pNumCols = rDetections.size();
    return;
  }

  *pNumRows = rTracks.size();
  *pNumCols = rDetections.size();
  for (int i = 0; i < *pNumRows; ++i) {
    std::vector<float> cost_matrix_row(*pNumCols);
    auto track_feature = rTracks[i]->mSmoothFeat;
    for (int j = 0; j < *pNumCols; ++j) {
      auto det_feature = rDetections[j].mCurrFeat;
      float feat_square = 0.0;
      for (int k = 0; k < det_feature.size(); ++k) {
        feat_square += (track_feature[k] - det_feature[k]) *
            (track_feature[k] - det_feature[k]);
      }
      cost_matrix_row[j] = std::sqrt(feat_square);
    }
    pCostMatrix->push_back(cost_matrix_row);
  }
}

void FuseMotion(
    const jde_kalman::KalmanFilter &rKalmanFilter,
    const std::vector<STrack*> &rTracks,
    const std::vector<STrack> &rDetections,
    std::vector<std::vector<float>> *pCostMatrix,
    bool onlyPosition,
    float coeff) {
  if (pCostMatrix->size() == 0) {
    return;
  }

  auto gating_dim = onlyPosition ? 2 : 4;
  float gating_threshold = rKalmanFilter.chi2inv95[gating_dim];

  std::vector<DetectBox> measurements(rDetections.size(), DetectBox());
  for (int i = 0; i < rDetections.size(); ++i) {
    auto xyah = rDetections[i].ToXyah();
    measurements[i][0] = xyah[0];
    measurements[i][1] = xyah[1];
    measurements[i][2] = xyah[2];
    measurements[i][3] = xyah[3];
  }

  for (int i = 0; i < rTracks.size(); ++i) {
    auto gating_distance = rKalmanFilter.GatingDistance(rTracks[i]->mMean,
        rTracks[i]->mCovariance, measurements, onlyPosition);

    for (int j = 0; j < (*pCostMatrix)[i].size(); ++j) {
      if (gating_distance[j] > gating_threshold) {
        (*pCostMatrix)[i][j] = FLT_MAX;
      }
      (*pCostMatrix)[i][j] = coeff * (*pCostMatrix)[i][j] + (1 - coeff) *
          gating_distance[j];
    }
  }
}

std::vector<std::vector<float>> IouDistance(
    const std::vector<STrack*> &rTracks1,
    const std::vector<STrack> &rTracks2,
    int *pNumRows,
    int *pNumCols) {
  *pNumRows = rTracks1.size();
  *pNumCols = rTracks2.size();
  std::vector<std::vector<float>> tlbrs_1(*pNumRows);
  std::vector<std::vector<float>> tlbrs_2(*pNumCols);
  for (int i = 0; i < *pNumRows; ++i) {
    tlbrs_1[i] = rTracks1[i]->mTlbr;
  }
  for (int i = 0; i < *pNumCols; ++i) {
    tlbrs_2[i] = rTracks2[i].mTlbr;
  }

  auto ious = Ious(tlbrs_1, tlbrs_2);
  std::vector<std::vector<float>> cost_matrix(ious.size());
  for (int i = 0; i < ious.size(); ++i) {
    cost_matrix[i].resize(ious[i].size());
    for (int j = 0; j < ious[i].size(); ++j) {
      cost_matrix[i][j] = 1 - ious[i][j];
    }
  }

  return cost_matrix;
}

std::vector<std::vector<float>> IouDistance(
    const std::vector<STrack> &rTracks1,
    const std::vector<STrack> &rTracks2) {
  std::vector<std::vector<float>> tlbrs_1(rTracks1.size());
  std::vector<std::vector<float>> tlbrs_2(rTracks2.size());
  for (int i = 0; i < rTracks1.size(); ++i) {
    tlbrs_1[i] = rTracks1[i].mTlbr;
  }
  for (int i = 0; i < rTracks2.size(); ++i) {
    tlbrs_2[i] = rTracks2[i].mTlbr;
  }

  auto ious = Ious(tlbrs_1, tlbrs_2);
  std::vector<std::vector<float>> cost_matrix(ious.size());
  for (int i = 0; i < ious.size(); ++i) {
    cost_matrix[i].resize(ious[i].size());
    for (int j = 0; j < ious[i].size(); ++j) {
      cost_matrix[i][j] = 1 - ious[i][j];
    }
  }

  return cost_matrix;
}

std::vector<std::vector<float>> Ious(
    const std::vector<std::vector<float>> &rTlbrs1,
    const std::vector<std::vector<float>> &rTlbrs2) {
  auto num_boxes = rTlbrs1.size();
  auto num_queries = rTlbrs2.size();
  std::vector<std::vector<float>> ious;
  if (num_boxes * num_queries == 0) {
    return ious;
  }

  ious.resize(num_boxes);
  for (int i = 0; i < num_boxes; ++i) {
    ious[i].resize(num_queries);
  }

  // bbox_ious
  for (int k = 0; k < num_queries; ++k) {
    auto box_area = (rTlbrs2[k][2] - rTlbrs2[k][0] + 1) *
        (rTlbrs2[k][3] - rTlbrs2[k][1] + 1);
    for (int n = 0; n < num_boxes; ++n) {
      auto inter_width = std::min(rTlbrs1[n][2], rTlbrs2[k][2]) -
          std::max(rTlbrs1[n][0], rTlbrs2[k][0]) + 1;
      if (inter_width > 0) {
        auto inter_height = std::min(rTlbrs1[n][3], rTlbrs2[k][3]) -
            std::max(rTlbrs1[n][1], rTlbrs2[k][1]) + 1;
        if (inter_height > 0) {
          auto inter_area = inter_width * inter_height;
          auto union_area = (rTlbrs1[n][2] - rTlbrs1[n][0] + 1) *
              (rTlbrs1[n][3] - rTlbrs1[n][1] + 1) + box_area - inter_area;
          ious[n][k] = inter_area / union_area;
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

void LinearAssignment(
    const std::vector<std::vector<float>> &rCostMatrix,
    int numRows,
    int numCols,
    float threshold,
    std::vector<std::vector<int>> *pMatches,
    std::vector<int> *pUnmatched1,
    std::vector<int> *pUnmatched2) {
  if (rCostMatrix.size() == 0) {
    for (int i = 0; i < numRows; ++i) {
      pUnmatched1->push_back(i);
    }
    for (int i = 0; i < numCols; ++i) {
      pUnmatched2->push_back(i);
    }
    return;
  }

  std::vector<int> rowsol;
  std::vector<int> colsol;
  auto c = jde_util::LapJv(rCostMatrix, &rowsol, &colsol, true, threshold);
  for (int i = 0; i < rowsol.size(); ++i) {
    if (rowsol[i] >= 0) {
      std::vector<int> match;
      match.push_back(i);
      match.push_back(rowsol[i]);
      pMatches->push_back(match);
    } else {
      pUnmatched1->push_back(i);
    }
  }

  for (int i = 0; i < colsol.size(); ++i) {
    if (colsol[i] < 0) {
      pUnmatched2->push_back(i);
    }
  }
}
}  // namespace matching
