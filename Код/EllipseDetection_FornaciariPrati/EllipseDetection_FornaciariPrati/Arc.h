#include <list>
#include <vector>
#include "opencv\cv.h"
#pragma once

using std::list;
using cv::Point;

class Arc
{
public:
	Arc(bool growsInTheRight, const Point& startPoint);
	// TODO: add copy constructor with move semantic, will increase perfomance
	void addToTheRight(const Point& point);
	void addToTheLeft(const Point& point);
	Point GetMostRightPoint() const;
	Point GetMostLeftPoint() const;
	bool IsGrowingInTheRight() const;
	void DrawArc(cv::Mat& canvas, uchar* color); // TODO: exclude it from Release version
	// TODO: probably add method for interpolation (smoothing)
private:
	list<Point> m_points;
	// For the sake of perfomance we never check that these actually are most left and most right points
	// Just be sure that you using correct function to add point in arc
	Point m_mostRightPoint, m_mostLeftPoint;
	// if true, this arc lies in II or IV quarter, if false - in I or III
	const bool m_growsInTheRight;
};

inline Point Arc::GetMostRightPoint() const
{
	return m_mostRightPoint;
}

inline Point Arc::GetMostLeftPoint() const
{
	return m_mostLeftPoint;
}

inline bool Arc::IsGrowingInTheRight() const
{
	return m_growsInTheRight;
}

typedef std::vector<Arc> Arcs;