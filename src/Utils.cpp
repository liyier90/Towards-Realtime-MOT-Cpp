#include "Utils.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "LapJv.h"

namespace jde_util {
cv::Scalar GetColor(int idx) {
  idx += 3;
  return cv::Scalar(37 * idx % 255, 17 * idx % 255, 29 * idx % 255);
}

cv::Size GetSize(
    int videoWidth,
    int videoHeight,
    int netWidth,
    int netHeight) {
  cv::Size size;
  auto width_ratio = static_cast<float>(netWidth) / videoWidth;
  auto height_ratio = static_cast<float>(netWidth) / videoHeight;
  auto ratio = std::min(width_ratio, height_ratio);
  size.width = static_cast<int>(videoWidth * ratio);
  size.height = static_cast<int>(videoHeight * ratio);

  return size;
}

double LapJv(
    const std::vector<std::vector<float>> &rCost,
    std::vector<int> *pRowsol,
    std::vector<int> *pColsol,
    bool extendCost,
    float costLimit,
    bool returnCost) {
  std::vector<std::vector<float>> cost_c;
  cost_c.assign(rCost.begin(), rCost.end());

  std::vector<std::vector<float> > cost_c_extended;

  int n_rows = cost_c.size();
  int n_cols = cost_c[0].size();
  pRowsol->resize(n_rows);
  pColsol->resize(n_cols);

  auto n = 0;
  if (n_rows == n_cols) {
    n = n_rows;
  } else if (!extendCost) {
    std::cout << "Square cost array expected. "
        << "If cost is intentionally non-square. "
        << "pass extendCost = true" << std::endl;
    system("pause");
    exit(0);
  }

  if (extendCost || costLimit < FLT_MAX) {
    n = n_rows + n_cols;
    cost_c_extended.resize(n);
    for (int i = 0; i < n; ++i) {
      cost_c_extended[i].resize(n);
    }

    if (costLimit < FLT_MAX) {
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          cost_c_extended[i][j] = costLimit / 2.0;
        }
      }
    } else {
      float max_cost = -1;
      for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
          if (cost_c[i][j] > max_cost) {
            max_cost = cost_c[i][j];
          }
        }
      }
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          cost_c_extended[i][j] = max_cost + 1;
        }
      }
    }

    for (int i = n_rows; i < n; ++i) {
      for (int j = n_cols; j < n; ++j) {
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

  double **p_cost;
  p_cost = new double *[sizeof(double *) * n];
  for (int i = 0; i < n; ++i) {
    p_cost[i] = new double[sizeof(double) * n];
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      p_cost[i][j] = cost_c[i][j];
    }
  }

  int *p_x_c = new int[sizeof(int) * n];
  int *p_y_c = new int[sizeof(int) * n];

  auto ret = lapjv_internal(n, p_cost, p_x_c, p_y_c);
  if (ret != 0) {
    std::cout << "Calculate Wrong!" << std::endl;
    system("pause");
    exit(0);
  }

  auto opt = 0.0;
  if (n != n_rows) {
    for (int i = 0; i < n; ++i) {
      if (p_x_c[i] >= n_cols) {
        p_x_c[i] = -1;
      }
      if (p_y_c[i] >= n_rows) {
        p_y_c[i] = -1;
      }
    }
    for (int i = 0; i < n_rows; ++i) {
      (*pRowsol)[i] = p_x_c[i];
    }
    for (int i = 0; i < n_cols; i++) {
      (*pColsol)[i] = p_y_c[i];
    }

    if (returnCost) {
      for (int i = 0; i < n_rows; ++i) {
        if ((*pRowsol)[i] != -1) {
          opt += p_cost[i][(*pRowsol)[i]];
        }
      }
    }
  } else if (returnCost) {
    for (int i = 0; i < n_rows; ++i) {
      opt += p_cost[i][(*pRowsol)[i]];
    }
  }

  for (int i = 0; i < n; ++i) {
    delete[] p_cost[i];
  }
  delete[] p_cost;
  delete[] p_x_c;
  delete[] p_y_c;

  return opt;
}

cv::Mat Letterbox(
    cv::Mat image,
    int height,
    int width) {
  auto shape = image.size();
  auto ratio = std::min(static_cast<float>(height) / shape.height,
      static_cast<float>(width) / shape.width);
  auto new_shape = cv::Size(round(shape.width * ratio),
      round(shape.height * ratio));
  auto width_padding = static_cast<float>(width - new_shape.width) / 2;
  auto height_padding = static_cast<float>(height - new_shape.height) / 2;
  int top = round(height_padding - 0.1);
  int bottom = round(height_padding + 0.1);
  int left = round(width_padding - 0.1);
  int right = round(width_padding + 0.1);

  cv::resize(image, image, new_shape, cv::INTER_AREA);
  cv::copyMakeBorder(image, image, top, bottom, left, right,
      cv::BORDER_CONSTANT, cv::Scalar(127.5, 127.5, 127.5));
  return image;
}

torch::Tensor Nms(
    const torch::Tensor &rBoxes,
    const torch::Tensor &rScores,
    float nmsThreshold) {
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

  auto *p_keep = keep.data_ptr<int64_t>();
  auto *p_suppressed = suppressed.data_ptr<uint8_t>();
  auto *p_order = order.data_ptr<int64_t>();
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

torch::Tensor NonMaxSuppression(
    torch::Tensor prediction,
    float nmsThreshold) {
  prediction.slice(1, 0, 4) = XywhToTlbr(prediction.slice(1, 0, 4));
  torch::Tensor nms_indices = Nms(prediction.slice(1, 0, 4),
      prediction.select(1, 4), nmsThreshold);

  return prediction.index_select(0, nms_indices);
}

void ScaleCoords(
    cv::Size imgSize,
    cv::Size origImgSize,
    torch::Tensor *pCoords) {
  auto gain_w = static_cast<float>(imgSize.width) / origImgSize.width;
  auto gain_h = static_cast<float>(imgSize.height) / origImgSize.height;
  auto gain = std::min(gain_w, gain_h);
  float pad_x = (imgSize.width - origImgSize.width * gain) / 2;
  float pad_y = (imgSize.height - origImgSize.height * gain) / 2;
  pCoords->select(1, 0) -= pad_x;
  pCoords->select(1, 1) -= pad_y;
  pCoords->select(1, 2) -= pad_x;
  pCoords->select(1, 3) -= pad_y;
  *pCoords /= gain;
  *pCoords = torch::clamp(*pCoords, 0);
}

torch::Tensor XywhToTlbr(torch::Tensor x) {
  auto y = torch::zeros_like(x);
  y.slice(1, 0, 1) = x.slice(1, 0, 1) - x.slice(1, 2, 3) / 2;
  y.slice(1, 1, 2) = x.slice(1, 1, 2) - x.slice(1, 3, 4) / 2;
  y.slice(1, 2, 3) = x.slice(1, 0, 1) + x.slice(1, 2, 3) / 2;
  y.slice(1, 3, 4) = x.slice(1, 1, 2) + x.slice(1, 3, 4) / 2;

  return y;
}
}  // namespace jde_util

