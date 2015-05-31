#include "opencv\cv.h"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include <string>
#include <vector>
#include <list>
#include "EllipseDetectionLibrary.h"

#ifdef WIN32
#define ourImread(filename, isColor) cvLoadImage(filename.c_str(), isColor)
#else
#define ourImread(filename, isColor) imread(filename, isColor)
#endif

using namespace cv;
using std::string;
using std::list;
using std::cout; using std::endl;

Mat src;
// дуги, относящиеся к разным четвертям коорд. плоскости, хранятся в разных векторах
vector<list<Point>> arcs[4];
vector<std::list<Point>::reverse_iterator> arcsMidPoints[4];
class PossibleTriplet{
public:
	int I, II, III, IV;
	PossibleTriplet(int aI, int aII, int aIII, int aIV) : I(aI), II(aII), III(aIII), IV(aIV) {}
};
vector<PossibleTriplet> possibleTriplets;

Mat findArcs(const Mat&);
void drawArc(Mat&, const list<Point>&, uchar* color);
void choosePossibleTriplets();
void findMidPoints();
Mat parallelsTest();
template <class Edge_Iterator, class Middle_Iterator>
Point3d findLineCrossingChords(Edge_Iterator edge, Middle_Iterator middle, Edge_Iterator edgeArcBorder, Middle_Iterator middleArcBorder);
Point2d getMainDirection(const vector<Point> &midPoints);

Point findCenterIandII(int arcI, int arcII, bool &errorFlag);
Point findCenterIIandIII(int arcII, int arcIII, bool &errorFlag);
Point findCenterIIIandIV(int arcIII, int arcI, bool &errorFlag);
Point findCenterIVandI(int arcIV, int arcI, bool &errorFlag);

int main(){
	//src = ourImread(string("C:\\Диплом\\Images\\centerAt421_306.bmp"), CV_LOAD_IMAGE_GRAYSCALE);
	//src = ourImread(string("C:\\Диплом\\Images\\тестовый круг.bmp"), CV_LOAD_IMAGE_GRAYSCALE);
	src = ourImread(string("C:\\Диплом\\Images\\test.bmp"), CV_LOAD_IMAGE_GRAYSCALE);
	EllipseDetector* detector = new FornaciariPratiDetector(4, 0.2);
	detector->DetectEllipses(src);

	return 0;
}