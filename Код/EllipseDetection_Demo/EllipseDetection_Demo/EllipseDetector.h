#include <string>
#include <vector>
#include "Ellipse.h"
#pragma once

using std::string;
using std::vector;

class EllipseDetector
{
public:
	virtual vector<Ellipse> DetectEllipses(const cv::Mat& src) = 0;
	virtual vector<Ellipse> DetailedEllipseDetection(const cv::Mat& src) = 0;
	EllipseDetector(string configFile="") {}
};