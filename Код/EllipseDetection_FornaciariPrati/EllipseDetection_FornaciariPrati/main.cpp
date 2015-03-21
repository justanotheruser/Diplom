#include "opencv\cv.h"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include <string>
#include <vector>
#include <list>
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
	src = ourImread(string("C:\\Диплом\\Images\\глаз.jpg"), CV_LOAD_IMAGE_GRAYSCALE);
	EllipseDetector* detector = new FornaciariPratiDetector();
	detector->DetailedEllipseDetection(src);





	//namedWindow("Source image", CV_WINDOW_AUTOSIZE);
	//imshow("Source image", src);

	/*Ellipse elps1(cvPoint(100, 50), 45, cvSize(60, 39));
	elps1.DrawOnImage(src, cvScalar(48), 2);
	imshow("Source image", src);*/

	/*Mat arcs_picture = findArcs(src);
	namedWindow("Arcs", CV_WINDOW_AUTOSIZE);
	imshow("Arcs", arcs_picture);*/

	/*choosePossibleTriplets();

	Mat paralleltsTestPicture = parallelsTest();
	namedWindow("Parallels test", CV_WINDOW_AUTOSIZE);
	imshow("Parallels test", paralleltsTestPicture);
	*/
	
	return 0;
}
void drawArc(Mat& canvas, const list<Point>& arc, uchar* color){
	for(auto i = arc.begin(); i != arc.end(); i++){
		canvas.at<cv::Vec3b>(*i)[0] = color[0];
		canvas.at<cv::Vec3b>(*i)[1] = color[1];
		canvas.at<cv::Vec3b>(*i)[2] = color[2];
	}
}
Mat findArcs(const Mat& src){
	Mat blurred, sobelX, sobelY, canny;
	int cannyLowTreshold = 50, blurKernelSize = 3, sobelKernelSize = 3;
	double blurSigma = 1, cannyHighLowtRatio = 2.5;

void choosePossibleTriplets(){
	// сначала выбираем совместные пары
	vector<Point> possibleIandII, possibleIIandIII, possibleIIIandIV, possibleIVandI;
	for(int aI = 0; aI < arcs[0].size(); aI++){ 
		for(int aII = 0; aII < arcs[1].size(); aII++){
			if(arcs[1][aII].front().x <= arcs[0][aI].back().x) // aI должна быть правее aII
				possibleIandII.push_back(Point(aI, aII));
		}
	}
	for(int aII = 0; aII < arcs[1].size(); aII++){
		for(int aIII = 0; aIII < arcs[2].size(); aIII++){
			if(arcs[2][aIII].back().y >= arcs[1][aII].back().y) // aIII должна быть под aII
				possibleIIandIII.push_back(Point(aII, aIII));
		}
	}

	for(int aIII = 0; aIII < arcs[2].size(); aIII++){ 
		for(int aIV = 0; aIV < arcs[3].size(); aIV++){
			if(arcs[2][aIII].front().x <= arcs[3][aIV].back().x) // aIII должна быть левее aIV
				possibleIIIandIV.push_back(Point(aIII, aIV));
		}
	}
	for(int aIV = 0; aIV < arcs[3].size(); aIV++){
		for(int aI = 0; aI < arcs[0].size(); aI++){
			if(arcs[3][aIV].front().y >= arcs[0][aI].front().y) // aIV должна быть ниже aI
				possibleIVandI.push_back(Point(aIV, aI));
		}
	}
	// теперь составляем возможные тройки
	for(int i = 0; i < possibleIandII.size(); i++){
		for(int j = 0; j < possibleIIandIII.size(); j++){
			// если сегмент из arcs[1] у этих пар один и тот же, то это возможная тройка
			if(possibleIandII[i].y == possibleIIandIII[j].x)
				possibleTriplets.push_back(PossibleTriplet(possibleIandII[i].x, possibleIandII[i].y, possibleIIandIII[j].y, -1));
		}
	}
	for(int i = 0; i < possibleIIandIII.size(); i++){
		for(int j = 0; j < possibleIIIandIV.size(); j++){
			if(possibleIIandIII[i].y == possibleIIIandIV[j].x)
				possibleTriplets.push_back(PossibleTriplet(-1, possibleIIandIII[i].x, possibleIIandIII[i].y, possibleIIIandIV[j].y));
		}
	}
	for(int i = 0; i < possibleIIIandIV.size(); i++){
		for(int j = 0; j < possibleIVandI.size(); j++){
			if(possibleIIIandIV[i].y == possibleIVandI[j].x)
				possibleTriplets.push_back(PossibleTriplet(possibleIVandI[j].y, -1, possibleIIIandIV[i].x, possibleIIIandIV[i].y));
		}
	}
	for(int i = 0; i < possibleIVandI.size(); i++){
		for(int j = 0; j < possibleIandII.size(); j++){
			if(possibleIVandI[i].y == possibleIandII[j].x)
				possibleTriplets.push_back(PossibleTriplet(possibleIandII[j].x, possibleIandII[j].y, -1, possibleIVandI[i].x));
		}
	}
	std::cout << arcs[0].size() * arcs[1].size() * arcs[2].size() + 
				 arcs[1].size() * arcs[2].size() * arcs[3].size() +
				 arcs[2].size() * arcs[3].size() * arcs[0].size() + 
				 arcs[3].size() * arcs[0].size() * arcs[1].size() 
			  << std::endl;
	std::cout << possibleTriplets.size() << std::endl;
}

void findMidPoints(){
	// заранее ищем середины кривых и сохраняем их	
	arcsMidPoints[0].reserve(arcs[0].size());	arcsMidPoints[1].reserve(arcs[1].size());
	arcsMidPoints[2].reserve(arcs[2].size());	arcsMidPoints[3].reserve(arcs[3].size());
	for(int arcType = 0; arcType < 4; arcType++){ // цикл по типам кривых
		for (int i = 0; i < arcs[arcType].size(); i++){ // цикл по кривым одного типа
			int number = 0, middle_number = arcs[arcType][i].size() / 2;
			for(auto point = arcs[arcType][i].rbegin(); point != arcs[arcType][i].rend(); point++, number++){ // цикл по точкам кривой
				if(number == middle_number){
					arcsMidPoints[arcType].push_back(point);
					break;
				}
			}
		}
	}
}

Mat parallelsTest(){
	const double thresholdCenterDiff = 5;
	findMidPoints(); // заранее ищем середины кривых для ускорения
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
// функция возвращает коэффициенту уравнения прямой
template <class Edge_Iterator, class Middle_Iterator>
Point3d findLineCrossingChords(Edge_Iterator edge, Middle_Iterator middle, Edge_Iterator edgeArcBorder, Middle_Iterator middleArcBorder){
	const int STEPS = 2;
	vector<Point> chordMidPoints; // содержит центры параллельных хорд, пересекающих две дуги
	Point dir_vector = *middle - *edge; // направляющий вектор исходной хорды
	while(true){
		for(int n = 0; n < STEPS && middle != middleArcBorder; n++, middle++);
		if(middle == middleArcBorder)
			break;
		// коэффициент из уравнения прямой
		int C = middle->y * dir_vector.x - middle->x * dir_vector.y;
		// ищем точку на второй дуге, которую пересекает эта прямая
		// будем брать ту, которая после подстановки в уравнение прямой с данным коэфеициентом С
		// и данным направляющим вектором даёт наименьшую невязку (отклонение от 0)
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
		// здесь находим середину хорды и запихиваем её в массив этих середин для дальнейшей обработки
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
	list<Point>::reverse_iterator edgeII = arcs[1][arcII].rbegin(); // левая нижняя точка II дуги
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
Point findCenterIIandIII(int arcII, int arcIII, bool &errorFlag){
	Point center;
	list<Point>::iterator edgeII = arcs[1][arcII].begin(); // верхняя правая точка
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
	list<Point>::reverse_iterator edgeIV = arcs[3][arcIV].rbegin(); // левая нижняя точка IV дуги
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
	list<Point>::reverse_iterator edgeIV = arcs[3][arcIV].rbegin(); // левая нижняя точка IV дуги
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