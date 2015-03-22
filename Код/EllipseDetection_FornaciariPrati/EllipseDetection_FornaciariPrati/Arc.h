#include <list>
#include <vector>
#include "opencv\cv.h"
#pragma once

using std::list;
using cv::Point;

class Arc
{
public:
	Arc(const Point& startPoint);
	// TODO: add copy constructor with move semantic, will increase perfomance
	void AddToTheRight(const Point& point); // TODO: добавить move-семантику
	void AddToTheLeft(const Point& point);
	Point GetMostRightPoint() const;
	Point GetMostLeftPoint() const;
	int Size() const;
	void DrawArc(cv::Mat& canvas, uchar* color); // TODO: exclude it from Release version
	// TODO: probably add method for interpolation (smoothing)
private:
	list<Point> m_points;
	// For the sake of perfomance we never check that these actually are most left and most right points
	// Just be sure that you using correct function to add point in arc
	Point m_mostRightPoint, m_mostLeftPoint;
};

inline Point Arc::GetMostRightPoint() const
{
	return m_mostRightPoint;
}

inline Point Arc::GetMostLeftPoint() const
{
	return m_mostLeftPoint;
}

inline int Arc::Size() const
{
	return m_points.size();
}
typedef std::vector<Arc> Arcs;