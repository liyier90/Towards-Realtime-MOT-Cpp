#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <opencv2/opencv.hpp>

#include "JDETracker.h"
#include "Utils.h"

int main() {
  std::string model_path = "../jit_convert/jde_576x320_torch14_gpu.pt";
  JDETracker tracker(model_path);

  std::string video_path = "../video/AVG-TownCentre.mp4";
  std::string output_dir = "../results";

  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    return 1;
  }
  auto num_frames = 0;
  std::chrono::duration<double> total_elapsed(0);
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

    auto padded_image = tracker.Preprocess(image);
    auto start_time = std::chrono::high_resolution_clock::now();
    auto stracks = tracker.Update(padded_image, image);
    auto end_time = std::chrono::high_resolution_clock::now();

    std::stringstream output_path_stream;
    output_path_stream << output_dir << "/" << std::setfill('0') << std::setw(5)
        << num_frames << ".jpg";
    jde_util::Visualize(image, stracks, num_frames);
    cv::imwrite(output_path_stream.str(), image);

    std::chrono::duration<double> elapsed = end_time - start_time;
    total_elapsed += elapsed;
    ++num_frames;
    if (num_frames % 10 == 0) {
      std::cout << 10.0 / total_elapsed.count() << " fps" << std::endl;
      total_elapsed = std::chrono::duration<double>::zero();
    }

    if (cv::waitKey(1) > 0) {
      break;
    }
  }
  cap.release();
  std::stringstream cmd_stream;
  cmd_stream << "ffmpeg -f image2 -i " << output_dir << "/%05d.jpg -c:v copy "
      << output_dir << "/results.mp4";
  std::system(cmd_stream.str().c_str());
  return 0;
}
