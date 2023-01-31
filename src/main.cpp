#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "JDETracker.h"
#include "Utils.h"

int main() {
  std::string model_path = "../jit_convert/jde_576x320_torch14_gpu.pt";
  JDETracker tracker(model_path);

  std::string video_path = "../video/AVG-TownCentre.mp4";

  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    return 1;
  }
  cv::Mat image;
  while (true) {
    cap >> image;
    if (image.empty()) {
      break;
    }
    int vh = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int vw = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    auto size = jde_util::GetSize(vw, vh, tracker.mNetWidth,
        tracker.mNetHeight);
    cv::resize(image, image, size);

    tracker.Update(image);

    if (cv::waitKey(1) > 0) {
      break;
    }
  }
  cap.release();

  return 0;
}
