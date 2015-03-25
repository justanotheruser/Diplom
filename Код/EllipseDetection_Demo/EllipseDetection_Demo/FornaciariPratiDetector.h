#include "EllipseDetector.h"
#include <list>
//#include "Arc.h"
#pragma once

using cv::Mat;
using cv::Point;

typedef std::list<Point> Arc;
typedef std::vector<Arc> Arcs;

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
	void heuristicSearchOfArcs();
	
	bool isEdgePoint(const Point& point);
	Arc findArcThatIncludesPoint(int x, int y, short* sX, short* sY);
	Mat findArcs(const Mat& src);

private:
	Mat m_sobelX, m_sobelY; // should use CS_16S
	Mat m_canny;
	Mat m_blurred; // TODO: delete it after implementing my own canny detector
};