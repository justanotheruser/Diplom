#pragma once
#include "opencv\cv.h"
#include <vector>
#include <map>
#include <fstream>
#include "Ellipse.h"

using cv::Point;
using cv::Size;
using cv::Mat;
using std::vector;

Point findArcMiddlePoint(const std::vector<Point>& arc);

void saveEllipses(const std::vector<Ellipse>& ellipses, const std::string& imgName, const Size& imgSize);

void loadEllipses(std::map<std::string, std::pair<Size, std::vector<Ellipse>>>& ellipses);