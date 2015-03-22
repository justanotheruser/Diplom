#include "Arc.h"

Arc::Arc(const Point& startPoint) 
{
	m_points.emplace_back(startPoint);
	m_mostRightPoint = m_mostLeftPoint = startPoint;
}

void Arc::AddToTheRight(const Point& point)
{
	m_points.push_front(point);
	m_mostRightPoint = point;
}

void Arc::AddToTheLeft(const Point& point)
{
	m_points.push_back(point);
	m_mostLeftPoint = point;
}

void Arc::DrawArc(cv::Mat& canvas, uchar* color){
	for(auto point : m_points){
		canvas.at<cv::Vec3b>(point)[0] = color[0];
		canvas.at<cv::Vec3b>(point)[1] = color[1];
		canvas.at<cv::Vec3b>(point)[2] = color[2];
	}
}