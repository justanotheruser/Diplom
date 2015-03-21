#include "EllipseDetector.h"
#pragma once

using cv::Mat;

class FornaciariPratiDetector : public EllipseDetector
{
public:
	FornaciariPratiDetector(string configFile="") {}
	// EllipseDetector functions
	virtual vector<Ellipse> DetectEllipses(const Mat& src);
	virtual vector<Ellipse> DetailedEllipseDetection(const Mat& src);
private:
	void getSobelDerivatives(const Mat& src);
	void useCannyDetector();
	Mat findArcs(const Mat& src);

private:
	Mat m_sobelX;
	Mat m_sobelY;
	Mat m_canny;
	Mat m_blurred; // TODO: delete it after implementing my own canny detector
};