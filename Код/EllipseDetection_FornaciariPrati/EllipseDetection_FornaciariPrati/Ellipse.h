#include "opencv\cv.h"
#include "opencv2\imgproc\imgproc.hpp"
#pragma once

class Ellipse
{
public:
	Ellipse(CvPoint center, double angle, CvSize axes);
	inline 	CvPoint GetCenter() const { return m_center; }
	inline 	double GetAngle() const { return m_angle; }
	inline 	CvSize GetAxes() const { return m_axes; }
	void DrawOnImage(cv::Mat& img, CvScalar color, int thickness=1, int line_type=8) const;
private:
	CvPoint m_center; 
	double m_angle; // half of the size of the ellipse main axes
	CvSize m_axes;
};