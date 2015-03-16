#include "EllipseDetector.h"
#pragma once

class FornaciariPratiDetector : public EllipseDetector
{
public:
	FornaciariPratiDetector(string configFile="") {}
	// EllipseDetector functions
	virtual vector<Ellipse> DetectEllipses(const cv::Mat& src);
	virtual vector<Ellipse> DetailedEllipseDetection(const cv::Mat& src);
private:
	cv::Mat findArcs(const cv::Mat& src);
};