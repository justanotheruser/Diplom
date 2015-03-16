#include "Ellipse.h"

Ellipse::Ellipse(CvPoint center, double angle, CvSize axes) 
	: m_center(center)
	, m_angle(angle)
	, m_axes(axes) {}


void Ellipse::DrawOnImage(cv::Mat& img, CvScalar color, int thickness, int line_type) const
{
	cv::ellipse(img, m_center, m_axes, m_angle, 0, 360, color, thickness, line_type);
}