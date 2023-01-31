#ifndef INCLUDE_MATCHING_H_
#define INCLUDE_MATCHING_H_

#include <vector>

#include "KalmanFilter.h"
#include "STrack.h"

namespace matching {
void EmbeddingDistance(
    const std::vector<STrack*> &rTracks,
    const std::vector<STrack> &rDetections,
    std::vector<std::vector<float>> *pCostMatrix,
    int *pNumRows,
    int *pNumCols);

void FuseMotion(
    const jde_kalman::KalmanFilter &rKalmanFilter,
    const std::vector<STrack*> &rTracks,
    const std::vector<STrack> &rDetections,
    std::vector<std::vector<float>> *pCostMatrix,
    bool onlyPosition = false,
    float coeff = 0.98);

std::vector<std::vector<float>> IouDistance(
    const std::vector<STrack*> &rTracks1,
    const std::vector<STrack> &rTracks2,
    int *pNumRows,
    int *pNumCols);

std::vector<std::vector<float>> IouDistance(
    const std::vector<STrack> &rTracks1,
    const std::vector<STrack> &rTracks2);

std::vector<std::vector<float>> Ious(
    const std::vector<std::vector<float>> &rTlbrs1,
    const std::vector<std::vector<float>> &rTlbrs2);

void LinearAssignment(
    const std::vector<std::vector<float>> &rCostMatrix,
    int numRows,
    int numCols,
    float threshhold,
    std::vector<std::vector<int>> *pMatches,
    std::vector<int> *pUnmatched1,
    std::vector<int> *pUnmatched2);
}  // namespace matching

#endif  // INCLUDE_MATCHING_H_
