#ifndef INCLUDE_UTILS_H_
#define INCLUDE_UTILS_H_

#include "STrack.h"
#include <cfloat>
#include <vector>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>  // NOLINT


namespace jde_util {
cv::Scalar GetColor(int idx);

cv::Size GetSize(
    int videoWidth,
    int videoHeight,
    int netWidth,
    int netHeight);

double LapJv(
    const std::vector<std::vector<float>> &rCost,
    std::vector<int> *pRowsol,
    std::vector<int> *pColsol,
    bool extendCost = false,
    float costLimit = FLT_MAX,
    bool returnCost = true);

cv::Mat Letterbox(
    cv::Mat img,
    int height,
    int width);

torch::Tensor Nms(
    const torch::Tensor &boxes,
    const torch::Tensor &scores,
    float overlap);

torch::Tensor NonMaxSuppression(
    torch::Tensor prediction,
    float nmsThreshold);

void ScaleCoords(
    cv::Size imgSize,
    cv::Size origImgSize,
    torch::Tensor *pCoords);

void Visualize(
    cv::Mat image,
    const std::vector<STrack> &rStracks);

torch::Tensor XywhToTlbr(torch::Tensor x);
}  // namespace jde_util

#endif  // INCLUDE_UTILS_H_

