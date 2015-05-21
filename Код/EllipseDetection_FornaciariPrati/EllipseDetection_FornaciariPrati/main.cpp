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
// дуги, относ€щиес€ к разным четверт€м коорд. плоскости, хран€тс€ в разных векторах
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
	src = ourImread(string("C:\\ƒиплом\\Images\\тестовый эллипс.bmp"), CV_LOAD_IMAGE_GRAYSCALE);
	EllipseDetector* detector = new FornaciariPratiDetector(4, 0.2);
	detector->DetectEllipses(src);

	return 0;
}
/*
Mat parallelsTest(){
	const double thresholdCenterDiff = 5;
	findMidPoints(); // заранее ищем середины кривых дл€ ускорени€
	Mat canvas = Mat::zeros(src.rows, src.cols, CV_8UC3);
	bool errorCenterFlag;
	uchar GREEN[3] = {0, 255, 0}, YELLOW[3] = {0, 255, 255}, RED[3] = {0, 0, 255}, BLUE[3] = {255, 0, 0};
	for(int triplet = 0; triplet < possibleTriplets.size(); triplet++){
		errorCenterFlag = false;
		if(possibleTriplets[triplet].I == -1){
			Point centerIIandIII = findCenterIIandIII(possibleTriplets[triplet].II, possibleTriplets[triplet].III, errorCenterFlag);
			Point centerIIIandIV = findCenterIIIandIV(possibleTriplets[triplet].III, possibleTriplets[triplet].IV, errorCenterFlag);
			if(errorCenterFlag)
				continue;
			Point diff = centerIIandIII - centerIIIandIV;
			if(norm(diff) < thresholdCenterDiff){
				drawArc(canvas, arcs[1][possibleTriplets[triplet].II], GREEN);
				drawArc(canvas, arcs[2][possibleTriplets[triplet].III], BLUE);
				drawArc(canvas, arcs[3][possibleTriplets[triplet].IV], YELLOW);
				ellipse(canvas, centerIIandIII, Size(5, 5), 0, 0, 360, Scalar(127, 127, 127));
				ellipse(canvas, centerIIIandIV, Size(5, 5), 0, 0, 360, Scalar(127, 127, 127));
			}	
		}
		else if(possibleTriplets[triplet].II == -1){
			Point centerIIIandIV = findCenterIIIandIV(possibleTriplets[triplet].III, possibleTriplets[triplet].IV, errorCenterFlag);
			Point centerIVandI = findCenterIVandI(possibleTriplets[triplet].IV, possibleTriplets[triplet].I, errorCenterFlag);
			if(errorCenterFlag)
				continue;
			Point diff = centerIVandI - centerIIIandIV;
			if(norm(diff) < thresholdCenterDiff){
				drawArc(canvas, arcs[0][possibleTriplets[triplet].I], RED);
				drawArc(canvas, arcs[2][possibleTriplets[triplet].III], BLUE);
				drawArc(canvas, arcs[3][possibleTriplets[triplet].IV], YELLOW);
				ellipse(canvas, centerIVandI, Size(5, 5), 0, 0, 360, Scalar(127, 127, 127));
				ellipse(canvas, centerIIIandIV, Size(5, 5), 0, 0, 360, Scalar(127, 127, 127));
			}
		}
		else if(possibleTriplets[triplet].III == -1){
			Point centerIandII = findCenterIandII(possibleTriplets[triplet].I, possibleTriplets[triplet].II, errorCenterFlag);
			Point centerIVandI = findCenterIVandI(possibleTriplets[triplet].IV, possibleTriplets[triplet].I, errorCenterFlag);
			if(errorCenterFlag)
				continue;
			Point diff = centerIVandI - centerIandII;
			if(norm(diff) < thresholdCenterDiff){
				drawArc(canvas, arcs[0][possibleTriplets[triplet].I], RED);
				drawArc(canvas, arcs[1][possibleTriplets[triplet].II], GREEN);
				drawArc(canvas, arcs[3][possibleTriplets[triplet].IV], YELLOW);
				ellipse(canvas, centerIVandI, Size(5, 5), 0, 0, 360, Scalar(127, 127, 127));
				ellipse(canvas, centerIandII, Size(5, 5), 0, 0, 360, Scalar(127, 127, 127));
			}
		}
		else{
			Point centerIandII = findCenterIandII(possibleTriplets[triplet].I, possibleTriplets[triplet].II, errorCenterFlag);
			Point centerIIandIII = findCenterIIandIII(possibleTriplets[triplet].II, possibleTriplets[triplet].III, errorCenterFlag);
			if(errorCenterFlag)
				continue;
			Point diff = centerIandII - centerIIandIII;
			if(norm(diff) < thresholdCenterDiff){
				drawArc(canvas, arcs[0][possibleTriplets[triplet].I], RED);
				drawArc(canvas, arcs[1][possibleTriplets[triplet].II], GREEN);
				drawArc(canvas, arcs[2][possibleTriplets[triplet].III], BLUE);
				ellipse(canvas, centerIIandIII, Size(5, 5), 0, 0, 360, Scalar(127, 127, 127));
				ellipse(canvas, centerIandII, Size(5, 5), 0, 0, 360, Scalar(127, 127, 127));
			}
		}
	}
	return canvas;
}*/

	/*
// функци€ возвращает коэффициенту уравнени€ пр€мой
template <class Edge_Iterator, class Middle_Iterator>
Point3d findLineCrossingChords(Edge_Iterator edge, Middle_Iterator middle, Edge_Iterator edgeArcBorder, Middle_Iterator middleArcBorder){
	const int STEPS = 2;
	vector<Point> chordMidPoints; // содержит центры параллельных хорд, пересекающих две дуги
	Point dir_vector = *middle - *edge; // направл€ющий вектор исходной хорды
	while(true){
		for(int n = 0; n < STEPS && middle != middleArcBorder; n++, middle++);
		if(middle == middleArcBorder)
			break;
		// коэффициент из уравнени€ пр€мой
		int C = middle->y * dir_vector.x - middle->x * dir_vector.y;
		// ищем точку на второй дуге, которую пересекает эта пр€ма€
		// будем брать ту, котора€ после подстановки в уравнение пр€мой с данным коэфеициентом —
		// и данным направл€ющим вектором даЄт наименьшую нев€зку (отклонение от 0)
		auto point = edge;
		int prev_discrepancy = abs(dir_vector.y * point->x - dir_vector.x * point->y + C);
		point++;
		for(; point != edgeArcBorder; point++){
			int discrepancy = abs(dir_vector.y * point->x - dir_vector.x * point->y + C);
			if(discrepancy > prev_discrepancy){
				point--;
				break;
			}
			prev_discrepancy = discrepancy;
		}
		// здесь находим середину хорды и запихиваем еЄ в массив этих середин дл€ дальнейшей обработки
		chordMidPoints.push_back((*point + *middle) * 0.5);
	}
	if(chordMidPoints.size() == 0){
		Point3d result(0,0,0); // невалидное значение
		return result;
	}
	Point2d p = getMainDirection(chordMidPoints);
	Point3d result(p.x, p.y, -p.x * chordMidPoints[0].x - p.y * chordMidPoints[0].y);
	return result;
}

Point2d getMainDirection(const vector<Point> &midPoints){
	int middle = midPoints.size() / 2;
	Point2d sum;
	for(int i = 0; i < middle; i++){
		Point2d diff = midPoints[i] - midPoints[middle + i];
		Point2d dir(diff.y, -diff.x);
		double dir_norm = norm(dir);
		dir.x /= dir_norm; dir.y /= dir_norm;
		sum += dir;
	}
	return sum;
}

Point findCenterIandII(int arcI, int arcII, bool &errorFlag){
	Point center;
	list<Point>::reverse_iterator edgeII = arcs[1][arcII].rbegin(); // лева€ нижн€€ точка II дуги
	list<Point>::iterator middleI = arcsMidPoints[0][arcI].base(); // середина I дуги
	list<Point>::iterator middleIArcBorder = arcs[0][arcI].end();
	list<Point>::reverse_iterator edgeIIArcBorder = arcs[1][arcII].rend();
	Point3d line1 = findLineCrossingChords<list<Point>::reverse_iterator,  list<Point>::iterator>(edgeII,  middleI, edgeIIArcBorder, middleIArcBorder);
	if(line1.x == 0 && line1.y == 0){
		errorFlag = true;
		return Point();
	}
	list<Point>::iterator edgeI = arcs[0][arcI].begin(); 
	list<Point>::reverse_iterator middleII = arcsMidPoints[1][arcII];
	list<Point>::reverse_iterator middleIIArcBorder = arcs[1][arcII].rend();
	list<Point>::iterator edgeIArcBorder = arcs[0][arcI].end();
	Point3d line2 = Point3d(findLineCrossingChords<list<Point>::iterator,  list<Point>::reverse_iterator>(edgeI,  middleII, edgeIArcBorder, middleIIArcBorder));
	if(line2.x == 0 && line2.y == 0){
		errorFlag = true;
		return Point();
	}
	if(line1.x != 0){
		center.y = (line2.z - line2.x/line1.x * line1.z) / (line2.x/line1.x * line1.y - line2.y);
		center.x = (-line1.z - line1.y*center.y) / line1.x;
	}
	else{
		center.y = -line1.z/line1.y;
		center.x = (-line2.z - line2.y*center.y) / line2.x;
	}
	return center;
}
/*
Point findCenterIIandIII(int arcII, int arcIII, bool &errorFlag){
	Point center;
	list<Point>::iterator edgeII = arcs[1][arcII].begin(); // верхн€€ права€ точка
	list<Point>::iterator middleIII = arcsMidPoints[2][arcIII].base();
	list<Point>::iterator middleIIIArcBorder = arcs[2][arcIII].end();
	list<Point>::iterator edgeIIArcBorder = arcs[1][arcII].end();
	Point3d line1 = findLineCrossingChords<list<Point>::iterator,  list<Point>::iterator>(edgeII,  middleIII, edgeIIArcBorder, middleIIIArcBorder);
	if(line1.x == 0 && line1.y == 0){
		errorFlag = true;
		return Point();
	}
	list<Point>::iterator edgeIII = arcs[2][arcIII].begin(); 
	list<Point>::iterator middleII = arcsMidPoints[1][arcII].base();
	list<Point>::iterator middleIIArcBorder = arcs[1][arcII].end();
	list<Point>::iterator edgeIIIArcBorder = arcs[2][arcIII].end();
	Point3d line2 = Point3d(findLineCrossingChords<list<Point>::iterator,  list<Point>::iterator>(edgeIII,  middleII, 
		edgeIIIArcBorder, middleIIArcBorder));
	if(line2.x == 0 && line2.y == 0){
		errorFlag = true;
		return Point();
	}
	if(line1.x != 0){
		center.y = (line2.z - line2.x/line1.x * line1.z) / (line2.x/line1.x * line1.y - line2.y);
		center.x = (-line1.z - line1.y*center.y) / line1.x;
	}
	else{
		center.y = -line1.z/line1.y;
		center.x = (-line2.z - line2.y*center.y) / line2.x;
	}
	return center;
}
Point findCenterIIIandIV(int arcIII, int arcIV, bool& errorFlag){
	Point center;
	list<Point>::reverse_iterator edgeIV = arcs[3][arcIV].rbegin(); // лева€ нижн€€ точка IV дуги
	list<Point>::iterator middleIII = arcsMidPoints[2][arcIII].base(); // середина III дуги
	list<Point>::iterator middleIIIArcBorder = arcs[2][arcIII].end();
	list<Point>::reverse_iterator edgeIVArcBorder = arcs[3][arcIV].rend();
	Point3d line1 = findLineCrossingChords<list<Point>::reverse_iterator,  list<Point>::iterator>(edgeIV,  middleIII, edgeIVArcBorder, middleIIIArcBorder);
	if(line1.x == 0 && line1.y == 0){
		errorFlag = true;
		return Point();
	}
	list<Point>::iterator edgeIII = arcs[2][arcIII].begin(); 
	list<Point>::reverse_iterator middleIV = arcsMidPoints[3][arcIV];
	list<Point>::reverse_iterator middleIVArcBorder = arcs[3][arcIV].rend();
	list<Point>::iterator edgeIIIArcBorder = arcs[2][arcIII].end();
	Point3d line2 = Point3d(findLineCrossingChords<list<Point>::iterator,  list<Point>::reverse_iterator>(edgeIII,  middleIV, edgeIIIArcBorder, middleIVArcBorder));
	if(line2.x == 0 && line2.y == 0){
		errorFlag = true;
		return Point();
	}
	if(line1.x != 0){
		center.y = (line2.z - line2.x/line1.x * line1.z) / (line2.x/line1.x * line1.y - line2.y);
		center.x = (-line1.z - line1.y*center.y) / line1.x;
	}
	else{
		center.y = -line1.z/line1.y;
		center.x = (-line2.z - line2.y*center.y) / line2.x;
	}
	return center;
}
Point findCenterIVandI(int arcIV, int arcI, bool& errorFlag){
	Point center;
	list<Point>::reverse_iterator edgeIV = arcs[3][arcIV].rbegin(); // лева€ нижн€€ точка IV дуги
	list<Point>::reverse_iterator middleI = arcsMidPoints[0][arcI]; // середина I дуги
	list<Point>::reverse_iterator middleIArcBorder = arcs[0][arcI].rend();
	list<Point>::reverse_iterator edgeIVArcBorder = arcs[3][arcIV].rend();
	Point3d line1 = findLineCrossingChords<list<Point>::reverse_iterator,  list<Point>::reverse_iterator>(edgeIV,  middleI, edgeIVArcBorder, middleIArcBorder);
	if(line1.x == 0 && line1.y == 0){
		errorFlag = true;
		return Point();
	}
	list<Point>::reverse_iterator edgeI = arcs[0][arcI].rbegin(); 
	list<Point>::reverse_iterator middleIV = arcsMidPoints[3][arcIV];
	list<Point>::reverse_iterator middleIVArcBorder = arcs[3][arcIV].rend();
	list<Point>::reverse_iterator edgeIArcBorder = arcs[0][arcI].rend();
	Point3d line2 = Point3d(findLineCrossingChords<list<Point>::reverse_iterator,  list<Point>::reverse_iterator>(edgeI,  middleIV, edgeIArcBorder, middleIVArcBorder));
	if(line2.x == 0 && line2.y == 0){
		errorFlag = true;
		return Point();
	}
	if(line1.x != 0){
		center.y = (line2.z - line2.x/line1.x * line1.z) / (line2.x/line1.x * line1.y - line2.y);
		center.x = (-line1.z - line1.y*center.y) / line1.x;
	}
	else{
		center.y = -line1.z/line1.y;
		center.x = (-line2.z - line2.y*center.y) / line2.x;
	}
	return center;
}*/