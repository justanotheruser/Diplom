#pragma once
#include "opencv\cv.h"
#include "opencv2\imgproc\imgproc.hpp"
#include <vector>

using cv::Point;
using cv::Size;
using cv::Scalar;
using std::vector;

struct Ellipse
{
	Ellipse(Point center, Size axes, double angle);
	inline 	CvPoint GetCenter() const { return m_center; }
	inline 	double GetAngle() const { return m_angle; }
	inline 	CvSize GetAxes() const { return m_axes; }
	void DrawOnImage(cv::Mat& img, Scalar color) const;
	vector<Point> FindEllipsePoints(Size img);
	vector<Point> GetEllipsePoints();

	Point m_center; 
	double m_angle;
	double m_angleInDegrees;
	double m_score1, m_score2;
	Size m_axes; // half of the size of the ellipse main axes
	vector<Point> m_ellipsePoints;
	void getEllipseReferencePoints(vector<Point>& referencePoints);
};