#pragma once
#include "opencv\cv.h"
#include <vector>

using cv::Point;
using cv::Size;
using cv::Mat;
using std::vector;

Point findArcMiddlePoint(const std::vector<Point>& arc);