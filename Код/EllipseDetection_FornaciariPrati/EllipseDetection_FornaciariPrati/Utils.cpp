#include "Utils.h"

void sincos( int angle, float& cosval, float& sinval )
{
    sinval = SinTable[angle];
    cosval = SinTable[450 - angle];
}

void ellipse2PolyOpt(Point center, Size axes, double angle, vector<Point>& pts)
{
	int _angle = cvRound(angle);
	while(_angle < 0)
        _angle += 360;
    while(_angle > 360)
        _angle -= 360;

    double size_a = axes.width, size_b = axes.height;
    double cx = center.x, cy = center.y;
	int delta = std::max(size_a, size_b);
	delta = delta < 3 ? 90 : delta < 10 ? 30 : delta < 15 ? 18 : 5;
	float alpha, beta;
    sincos(_angle, alpha, beta);
    pts.resize(0);
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
            pts.push_back(pt);
            prevPt = pt;
        }
    }
}

// http://en.wikipedia.org/wiki/Bresenham's_line_algorithm
void myLine(Mat& img, const Point& p0, const Point& p1, vector<Point>& ellipsePoints)
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
			if (x < 0 || x > img.size().width || y < 0 || y >= img.size().height)
				return;
			ellipsePoints.emplace_back(x, y);
			img.at<cv::Vec3b>(Point(x, y))[0] = 255;
			error += deltaerr;
			while (error >= 0.5)
			{
				if (x < 0 || x > img.size().width || y < 0 || y >= img.size().height)
					return;
				ellipsePoints.emplace_back(x, y);
				img.at<cv::Vec3b>(Point(x, y))[0] = 255;
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
			if (x < 0 || x > img.size().width || y < 0 || y >= img.size().height)
				return;
			ellipsePoints.emplace_back(x, y);
			img.at<cv::Vec3b>(Point(x, y))[0] = 255;
			error += deltaerr;
			while (error >= 0.5)
			{
				if (x < 0 || x > img.size().width || y < 0 || y >= img.size().height)
					return;
				ellipsePoints.emplace_back(x, y);
				img.at<cv::Vec3b>(Point(x, y))[0] = 255;
				x += deltaX;
				error -= 1.0;
			}
		}
	}
}

void polyLines2Points(Mat& img, const Point* v, int count, vector<Point>& ellipsePoints)
{
    int i = count - 1;
    Point p0;
    p0 = v[i];
    for( i = 0; i < count; i++ )
    {
        Point p = v[i];
		myLine(img, p, p0, ellipsePoints);
        p0 = p;
    }
}
