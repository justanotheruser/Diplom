#include <string>
#include <vector>
#include "Ellipse.h"
#pragma once

using std::string;
using std::vector;

class EllipseDetector
{
public:
	virtual vector<Ellipse> DetectEllipses(const cv::Mat& img) = 0;
	EllipseDetector(string configFile="") {}
};