/*
author: samylee
github: https://github.com/samylee
date: 08/19/2021
*/

#include <iostream>

#include "JDETracker.h"

int main()
{
  std::string model_path = "../jit_convert/jde_576x320_torch14_gpu.pt";
  JDETracker tracker(model_path);

  std::string video_path = "../video/AVG-TownCentre.mp4";
  tracker.Update(video_path);

  return 0;
}
