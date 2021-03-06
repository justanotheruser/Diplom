#include "Ellipse.h"
#define _USE_MATH_DEFINES
#include <math.h>

Ellipse::Ellipse(const Point& center, const Size& axes, double angle, const Size& imgSize) 
	: m_center(center), m_angle(angle), m_axes(axes)
{
	findEllipsePoints(imgSize);
}


void Ellipse::DrawOnImage(cv::Mat& img, Scalar color) const
{
	for (auto point : m_ellipsePoints)
	{
		img.at<cv::Vec3b>(point)[0] = color[0];
		img.at<cv::Vec3b>(point)[1] = color[1];
		img.at<cv::Vec3b>(point)[2] = color[2];
	}
}

static const float SinTable[] =
    { 0.0000000f, 0.0174524f, 0.0348995f, 0.0523360f, 0.0697565f, 0.0871557f,
    0.1045285f, 0.1218693f, 0.1391731f, 0.1564345f, 0.1736482f, 0.1908090f,
    0.2079117f, 0.2249511f, 0.2419219f, 0.2588190f, 0.2756374f, 0.2923717f,
    0.3090170f, 0.3255682f, 0.3420201f, 0.3583679f, 0.3746066f, 0.3907311f,
    0.4067366f, 0.4226183f, 0.4383711f, 0.4539905f, 0.4694716f, 0.4848096f,
    0.5000000f, 0.5150381f, 0.5299193f, 0.5446390f, 0.5591929f, 0.5735764f,
    0.5877853f, 0.6018150f, 0.6156615f, 0.6293204f, 0.6427876f, 0.6560590f,
    0.6691306f, 0.6819984f, 0.6946584f, 0.7071068f, 0.7193398f, 0.7313537f,
    0.7431448f, 0.7547096f, 0.7660444f, 0.7771460f, 0.7880108f, 0.7986355f,
    0.8090170f, 0.8191520f, 0.8290376f, 0.8386706f, 0.8480481f, 0.8571673f,
    0.8660254f, 0.8746197f, 0.8829476f, 0.8910065f, 0.8987940f, 0.9063078f,
    0.9135455f, 0.9205049f, 0.9271839f, 0.9335804f, 0.9396926f, 0.9455186f,
    0.9510565f, 0.9563048f, 0.9612617f, 0.9659258f, 0.9702957f, 0.9743701f,
    0.9781476f, 0.9816272f, 0.9848078f, 0.9876883f, 0.9902681f, 0.9925462f,
    0.9945219f, 0.9961947f, 0.9975641f, 0.9986295f, 0.9993908f, 0.9998477f,
    1.0000000f, 0.9998477f, 0.9993908f, 0.9986295f, 0.9975641f, 0.9961947f,
    0.9945219f, 0.9925462f, 0.9902681f, 0.9876883f, 0.9848078f, 0.9816272f,
    0.9781476f, 0.9743701f, 0.9702957f, 0.9659258f, 0.9612617f, 0.9563048f,
    0.9510565f, 0.9455186f, 0.9396926f, 0.9335804f, 0.9271839f, 0.9205049f,
    0.9135455f, 0.9063078f, 0.8987940f, 0.8910065f, 0.8829476f, 0.8746197f,
    0.8660254f, 0.8571673f, 0.8480481f, 0.8386706f, 0.8290376f, 0.8191520f,
    0.8090170f, 0.7986355f, 0.7880108f, 0.7771460f, 0.7660444f, 0.7547096f,
    0.7431448f, 0.7313537f, 0.7193398f, 0.7071068f, 0.6946584f, 0.6819984f,
    0.6691306f, 0.6560590f, 0.6427876f, 0.6293204f, 0.6156615f, 0.6018150f,
    0.5877853f, 0.5735764f, 0.5591929f, 0.5446390f, 0.5299193f, 0.5150381f,
    0.5000000f, 0.4848096f, 0.4694716f, 0.4539905f, 0.4383711f, 0.4226183f,
    0.4067366f, 0.3907311f, 0.3746066f, 0.3583679f, 0.3420201f, 0.3255682f,
    0.3090170f, 0.2923717f, 0.2756374f, 0.2588190f, 0.2419219f, 0.2249511f,
    0.2079117f, 0.1908090f, 0.1736482f, 0.1564345f, 0.1391731f, 0.1218693f,
    0.1045285f, 0.0871557f, 0.0697565f, 0.0523360f, 0.0348995f, 0.0174524f,
    0.0000000f, -0.0174524f, -0.0348995f, -0.0523360f, -0.0697565f, -0.0871557f,
    -0.1045285f, -0.1218693f, -0.1391731f, -0.1564345f, -0.1736482f, -0.1908090f,
    -0.2079117f, -0.2249511f, -0.2419219f, -0.2588190f, -0.2756374f, -0.2923717f,
    -0.3090170f, -0.3255682f, -0.3420201f, -0.3583679f, -0.3746066f, -0.3907311f,
    -0.4067366f, -0.4226183f, -0.4383711f, -0.4539905f, -0.4694716f, -0.4848096f,
    -0.5000000f, -0.5150381f, -0.5299193f, -0.5446390f, -0.5591929f, -0.5735764f,
    -0.5877853f, -0.6018150f, -0.6156615f, -0.6293204f, -0.6427876f, -0.6560590f,
    -0.6691306f, -0.6819984f, -0.6946584f, -0.7071068f, -0.7193398f, -0.7313537f,
    -0.7431448f, -0.7547096f, -0.7660444f, -0.7771460f, -0.7880108f, -0.7986355f,
    -0.8090170f, -0.8191520f, -0.8290376f, -0.8386706f, -0.8480481f, -0.8571673f,
    -0.8660254f, -0.8746197f, -0.8829476f, -0.8910065f, -0.8987940f, -0.9063078f,
    -0.9135455f, -0.9205049f, -0.9271839f, -0.9335804f, -0.9396926f, -0.9455186f,
    -0.9510565f, -0.9563048f, -0.9612617f, -0.9659258f, -0.9702957f, -0.9743701f,
    -0.9781476f, -0.9816272f, -0.9848078f, -0.9876883f, -0.9902681f, -0.9925462f,
    -0.9945219f, -0.9961947f, -0.9975641f, -0.9986295f, -0.9993908f, -0.9998477f,
    -1.0000000f, -0.9998477f, -0.9993908f, -0.9986295f, -0.9975641f, -0.9961947f,
    -0.9945219f, -0.9925462f, -0.9902681f, -0.9876883f, -0.9848078f, -0.9816272f,
    -0.9781476f, -0.9743701f, -0.9702957f, -0.9659258f, -0.9612617f, -0.9563048f,
    -0.9510565f, -0.9455186f, -0.9396926f, -0.9335804f, -0.9271839f, -0.9205049f,
    -0.9135455f, -0.9063078f, -0.8987940f, -0.8910065f, -0.8829476f, -0.8746197f,
    -0.8660254f, -0.8571673f, -0.8480481f, -0.8386706f, -0.8290376f, -0.8191520f,
    -0.8090170f, -0.7986355f, -0.7880108f, -0.7771460f, -0.7660444f, -0.7547096f,
    -0.7431448f, -0.7313537f, -0.7193398f, -0.7071068f, -0.6946584f, -0.6819984f,
    -0.6691306f, -0.6560590f, -0.6427876f, -0.6293204f, -0.6156615f, -0.6018150f,
    -0.5877853f, -0.5735764f, -0.5591929f, -0.5446390f, -0.5299193f, -0.5150381f,
    -0.5000000f, -0.4848096f, -0.4694716f, -0.4539905f, -0.4383711f, -0.4226183f,
    -0.4067366f, -0.3907311f, -0.3746066f, -0.3583679f, -0.3420201f, -0.3255682f,
    -0.3090170f, -0.2923717f, -0.2756374f, -0.2588190f, -0.2419219f, -0.2249511f,
    -0.2079117f, -0.1908090f, -0.1736482f, -0.1564345f, -0.1391731f, -0.1218693f,
    -0.1045285f, -0.0871557f, -0.0697565f, -0.0523360f, -0.0348995f, -0.0174524f,
    -0.0000000f, 0.0174524f, 0.0348995f, 0.0523360f, 0.0697565f, 0.0871557f,
    0.1045285f, 0.1218693f, 0.1391731f, 0.1564345f, 0.1736482f, 0.1908090f,
    0.2079117f, 0.2249511f, 0.2419219f, 0.2588190f, 0.2756374f, 0.2923717f,
    0.3090170f, 0.3255682f, 0.3420201f, 0.3583679f, 0.3746066f, 0.3907311f,
    0.4067366f, 0.4226183f, 0.4383711f, 0.4539905f, 0.4694716f, 0.4848096f,
    0.5000000f, 0.5150381f, 0.5299193f, 0.5446390f, 0.5591929f, 0.5735764f,
    0.5877853f, 0.6018150f, 0.6156615f, 0.6293204f, 0.6427876f, 0.6560590f,
    0.6691306f, 0.6819984f, 0.6946584f, 0.7071068f, 0.7193398f, 0.7313537f,
    0.7431448f, 0.7547096f, 0.7660444f, 0.7771460f, 0.7880108f, 0.7986355f,
    0.8090170f, 0.8191520f, 0.8290376f, 0.8386706f, 0.8480481f, 0.8571673f,
    0.8660254f, 0.8746197f, 0.8829476f, 0.8910065f, 0.8987940f, 0.9063078f,
    0.9135455f, 0.9205049f, 0.9271839f, 0.9335804f, 0.9396926f, 0.9455186f,
    0.9510565f, 0.9563048f, 0.9612617f, 0.9659258f, 0.9702957f, 0.9743701f,
    0.9781476f, 0.9816272f, 0.9848078f, 0.9876883f, 0.9902681f, 0.9925462f,
    0.9945219f, 0.9961947f, 0.9975641f, 0.9986295f, 0.9993908f, 0.9998477f,
    1.0000000f
};

void sincos( int angle, float& cosval, float& sinval )
{
    sinval = SinTable[angle];
    cosval = SinTable[450 - angle];
}

// ���������������� ������ ellipse2Poly, ���������� ������� OpenCV, � ������� ������
// � ����� ����� ������������� � ����� ���������
void Ellipse::getEllipseReferencePoints(vector<Point>& referencePoints)
{
	int _angle = cvRound(m_angle/M_PI * 180);
	while(_angle < 0)
        _angle += 360;
    while(_angle > 360)
        _angle -= 360;

    double size_a = m_axes.width, size_b = m_axes.height;
    double cx = m_center.x, cy = m_center.y;
	int delta = std::max(size_a, size_b);
	delta = delta < 3 ? 90 : delta < 10 ? 30 : delta < 15 ? 18 : 5;
	float alpha, beta;
    sincos(_angle, alpha, beta);
    referencePoints.resize(0);
	Point prevPt(INT_MIN,INT_MIN);

    for(int angle = 0; angle < 360 + delta; angle += delta )
    {
        double x, y;
        x = size_a * SinTable[450-angle];
        y = size_b * SinTable[angle];
        Point pt;
        pt.x = cvRound( cx + x * alpha - y * beta );
        pt.y = cvRound( cy + x * beta + y * alpha );
        if( pt != prevPt ){
            referencePoints.push_back(pt);
            prevPt = pt;
        }
    }
}

// http://en.wikipedia.org/wiki/Bresenham's_line_algorithm
void getPointsInLineBetween(const Point& p0, const Point& p1, Size imgSize, vector<Point>& pointsInLine)
{
	double dx = p1.x - p0.x;
    double dy = p1.y - p0.y;
    double error = 0;
	if (dx != 0)
	{
		double deltaerr = abs(dy / dx);
        int y = p0.y;
		int deltaX = p1.x > p0.x ? 1 : -1;
		int deltaY = p1.y > p0.y ? 1 : -1;
		for (int x = p0.x; x != p1.x; x+=deltaX)
		{
			if (x < 0 || x > imgSize.width || y < 0 || y >= imgSize.height)
				return;
			pointsInLine.emplace_back(x, y);
			error += deltaerr;
			while (error >= 0.5)
			{
				if (x < 0 || x > imgSize.width || y < 0 || y >= imgSize.height)
					return;
				pointsInLine.emplace_back(x, y);
				y += deltaY;
				error -= 1.0;
			}
		}
	}
	else if (dy != 0)
	{
		double deltaerr = abs(dx / dy);
        int x = p0.x;
		int deltaX = p1.x > p0.x ? 1 : -1;
		int deltaY = p1.y > p0.y ? 1 : -1;
		for (int y = p0.y; y != p1.y; y+=deltaY)
		{
			if (x < 0 || x > imgSize.width || y < 0 || y >= imgSize.height)
				return;
			pointsInLine.emplace_back(x, y);
			error += deltaerr;
			while (error >= 0.5)
			{
				if (x < 0 || x > imgSize.width || y < 0 || y >= imgSize.height)
					return;
				pointsInLine.emplace_back(x, y);
				x += deltaX;
				error -= 1.0;
			}
		}
	}
}

void Ellipse::findEllipsePoints(const Size& imgSize)
{
	m_ellipsePoints.resize(0);
    vector<Point> referencePoints;
	getEllipseReferencePoints(referencePoints);
	int i = referencePoints.size() - 1;
    Point p0;
    p0 = referencePoints[i];
	for(auto p : referencePoints)
    {
		getPointsInLineBetween(p, p0, imgSize, m_ellipsePoints);
        p0 = p;
    }
}

vector<Point> Ellipse::GetEllipsePoints()
{
	return m_ellipsePoints;
}

void Ellipse::SetImgSize(const Size& imgSize)
{
	findEllipsePoints(imgSize);
}

std::ostream & operator<<(std::ostream & os, const Ellipse& e)
{
	os << e.m_center.x << " " << e.m_center.y << " " << e.m_axes.width << " "
		<< e.m_axes.height << " " << e.m_angle << std::endl;
	return os;
}

std::istream & operator>>(std::istream & is, Ellipse& e)
{
	is >> e.m_center.x >> e.m_center.y >> e.m_axes.width 
	   >> e.m_axes.height >> e.m_angle;
	return is;
}