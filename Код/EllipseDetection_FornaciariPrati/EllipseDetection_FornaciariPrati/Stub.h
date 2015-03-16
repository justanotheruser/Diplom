#include "EllipseDetector.h"
#pragma once

class Stub : public EllipseDetector
{
public:
	Stub(string configFile) {}
	// EllipseDetector functions
	virtual vector<Ellipse> DetectEllipses(const cv::Mat& img);
};