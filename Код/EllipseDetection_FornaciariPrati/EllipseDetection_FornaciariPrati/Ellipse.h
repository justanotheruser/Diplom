#pragma once
#include "opencv\cv.h"
#include "opencv2\imgproc\imgproc.hpp"
#include <vector>

using cv::Point;
using cv::Size;
using cv::Scalar;
using std::vector;

class Ellipse
{
public:
	Ellipse(const Point& center, const Size& axes, double angle, const Size& imgSize);
	Ellipse() {};
	void DrawOnImage(cv::Mat& img, Scalar color) const;
	vector<Point> GetEllipsePoints();
	void SetImgSize(const Size& imgSize);
	inline void SetScore(double score) { m_score = score; }
	inline double GetScore() const { return m_score; }
	inline 	CvPoint GetCenter() const { return m_center; }
	inline 	double GetAngle() const { return m_angle; }
	inline 	CvSize GetAxes() const { return m_axes; }

	friend std::ostream & operator<<(std::ostream & os, const Ellipse& e);
	friend std::istream & operator>>(std::istream & is, Ellipse& e);
private:
	Point m_center; 
	double m_angle; // radians
	double m_score;
	Size m_axes; // half of the size of the ellipse main axes
	vector<Point> m_ellipsePoints;
	void getEllipseReferencePoints(vector<Point>& referencePoints);
	void findEllipsePoints(const Size& img);
};