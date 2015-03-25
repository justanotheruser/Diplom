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
	src = ourImread(string("C:\\Диплом\\\Images\\bicycle0.jpg"), CV_LOAD_IMAGE_GRAYSCALE);
	namedWindow("Source image", CV_WINDOW_AUTOSIZE);
	imshow("Source image", src);

	Mat arcs_picture = findArcs(src);
	namedWindow("Arcs", CV_WINDOW_AUTOSIZE);
	imshow("Arcs", arcs_picture);

	choosePossibleTriplets();

	Mat paralleltsTestPicture = parallelsTest();
	namedWindow("Parallels test", CV_WINDOW_AUTOSIZE);
	imshow("Parallels test", paralleltsTestPicture);
	waitKey(0);
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

	GaussianBlur(src, blurred, Size(blurKernelSize, blurKernelSize), blurSigma, blurSigma); 
	Sobel(blurred, sobelX, CV_16S, 1, 0, sobelKernelSize);
	Sobel(blurred, sobelY, CV_16S, 0, 1, sobelKernelSize);
	Canny(blurred, canny, cannyLowTreshold, cannyLowTreshold * cannyHighLowtRatio);
	namedWindow("Canny", CV_WINDOW_AUTOSIZE);
	imshow("Canny", canny);

	Mat result = Mat::zeros(canny.rows, canny.cols, CV_8UC3);
	// используется в двух местах для отсекания кривых, смахивающих на линии
	const int lineSimilarityThreshold = 20; // предельное число точек подряд, у которых одна из координат неизменна

	for(int row = 0; row < canny.rows; row++){ 
		uchar* p = canny.ptr(row);
		short* sX = sobelX.ptr<short>(row);
		short* sY = sobelY.ptr<short>(row);
		for(int col = 0; col < canny.cols; col++, p++, sX++, sY++) {
			if(*p == 255 && *sX != 0){
				// определяем направление градиента (отделяем  II и IV от I и III)
				int dx, dy; // смещения, прибавляемые к граничным точкам
				int yBorder, sYsXSign;
				if(*sY * *sX > 0){ // знак такой же, как и *sY / *sX, но умножение чуть быстрее
					dx = 1; dy = -1; yBorder = 0; sYsXSign = 1;
				}
				else if(*sY * *sX < 0){
					dx = 1; dy = 1; yBorder = canny.rows-1; sYsXSign = -1;
				}
				else
					continue;
				list<Point> new_arc;
				new_arc.push_front(Point(col, row));
				canny.at<uchar>(Point(col, row)) = 0;	

				// ищем все принадлежащие этой арке точки
				int rightX, rightY, leftX, leftY;
				int newRightX = col, newRightY = row;
				// переменные для контроля за кривизной
				int looksLikeVerticalLine = 0, looksLikeHorizontalLine = 0;
				do{  // ищем точки кривой вправо от исходной
					rightX = newRightX; rightY = newRightY;
					uchar* h = canny.ptr(rightY + dy); // точка по диагонали
					h += rightX + dx;
					if(*h == 255){
						short* new_sX = sobelX.ptr<short>(rightY + dy); // проверяем, что направление градиента в этой точке остаётся тем же
						new_sX += rightX + dx;
						short* new_sY = sobelY.ptr<short>(rightY + dy);
						new_sY += rightX + dx;
						if(sYsXSign * (*new_sY) * (*new_sX) > 0){ // знаки совпадают, всё хорошо
							newRightY = rightY + dy;
							newRightX = rightX + dx;
							looksLikeVerticalLine = 0;	looksLikeHorizontalLine = 0;
							new_arc.push_front(Point(newRightX, newRightY));
							canny.at<uchar>(Point(newRightX, newRightY)) = 0;
							continue;
						}
					}
					h = canny.ptr(rightY + dy); // изменился только y
					h += rightX;
					if(*h == 255){
						// проверяем, как давно менялся x
						looksLikeVerticalLine++;
						// если слишком давно, то наша кривая постепенно превращается в вертикальную прямую и нужно оборвать её
						// если нет, то продолжаем
						if(looksLikeVerticalLine < lineSimilarityThreshold){
							short* new_sX = sobelX.ptr<short>(rightY + dy); // проверяем, что направление градиента в этой точке остаётся тем же
							new_sX += rightX;
							short* new_sY = sobelY.ptr<short>(rightY + dy);
							new_sY += rightX;
							if(sYsXSign * (*new_sY) * (*new_sX) > 0){
								newRightY = rightY + dy;
								looksLikeHorizontalLine = 0;
								new_arc.push_front(Point(newRightX, newRightY));
								canny.at<uchar>(Point(newRightX, newRightY)) = 0;
								continue;
							}
						}
					}
					h = canny.ptr(rightY); // изменился только х
					h += rightX + dx;
					if(*h == 255){
						// проверяем, как давно менялся y
						looksLikeHorizontalLine++;
						if(looksLikeHorizontalLine < lineSimilarityThreshold){
							short* new_sX = sobelX.ptr<short>(rightY); // проверяем, что направление градиента в этой точке остаётся тем же
							new_sX += rightX + dx;
							short* new_sY = sobelY.ptr<short>(rightY);
							new_sY += rightX + dx;
							if(sYsXSign * (*new_sY) * (*new_sX) > 0){
								newRightX = rightX + dx;
								looksLikeVerticalLine = 0;
								new_arc.push_front(Point(newRightX, newRightY));
								canny.at<uchar>(Point(newRightX, newRightY)) = 0;
								continue;
							}
						}
					}
				// выполняем пока верхняя/правая граница не перестанет меняться или не упрёмся в край изображения
				}while(!(newRightX==rightX && newRightY==rightY) && newRightX != canny.cols-1 && newRightY != yBorder);

				yBorder = yBorder==0 ? canny.rows - 1 : 0;
				int newLeftX = col, newLeftY = row;
				looksLikeVerticalLine = 0; looksLikeHorizontalLine = 0;
				do{
					leftX = newLeftX; leftY = newLeftY;
					uchar* l = canny.ptr(leftY - dy); // точка по диагонали
					l += leftX - dx;
					if(*l == 255){
						short* new_sX = sobelX.ptr<short>(leftY - dy); // проверяем, что направление градиента в этой точке остаётся тем же
						new_sX += leftX - dx;
						short* new_sY = sobelY.ptr<short>(leftY - dy);
						new_sY += leftX - dx;
						if(sYsXSign * (*new_sY) * (*new_sX) > 0){ // знаки совпадают, всё хорошо
							newLeftX = leftX - dx;
							newLeftY = leftY - dy;
							looksLikeVerticalLine = 0;	looksLikeHorizontalLine = 0;
							new_arc.push_back(Point(newLeftX, newLeftY));
							canny.at<uchar>(Point(newLeftX, newLeftY)) = 0;
							continue;
						}
					}
					l = canny.ptr(leftY - dy); // изменился только y
					l += leftX;
					if(*l == 255){
						// проверяем, как давно менялся x
						looksLikeVerticalLine++;
						if(looksLikeVerticalLine < lineSimilarityThreshold){
							short* new_sX = sobelX.ptr<short>(leftY - dy); // проверяем, что направление градиента в этой точке остаётся тем же
							new_sX += leftX;
							short* new_sY = sobelY.ptr<short>(leftY - dy);
							new_sY += leftX;
							if(sYsXSign * (*new_sY) * (*new_sX) > 0){
								newLeftY = leftY - dy;
								looksLikeHorizontalLine = 0;
								new_arc.push_back(Point(newLeftX, newLeftY));
								canny.at<uchar>(Point(newLeftX, newLeftY)) = 0;
								continue;
							}
						}
					}


					l = canny.ptr(leftY); // изменился только х
					l += leftX - dx;
					if(*l == 255){
						// проверяем, как давно менялся y
						looksLikeHorizontalLine++;
						if(looksLikeHorizontalLine < lineSimilarityThreshold){
							short* new_sX = sobelX.ptr<short>(leftY); // проверяем, что направление градиента в этой точке остаётся тем же
							new_sX += leftX - dx;
							short* new_sY = sobelY.ptr<short>(leftY);
							new_sY += leftX - dx;
							if(sYsXSign * (*new_sY) * (*new_sX) > 0){
								newLeftX = leftX - dx;
								looksLikeVerticalLine = 0;
								new_arc.push_back(Point(newLeftX, newLeftY));
								canny.at<uchar>(Point(newLeftX, newLeftY)) = 0;
								continue;
							}
						}
					}
				// выполняем пока нижняя/левая граница не перестанет меняться или не упрёмся в край изображения
				}while(!(newLeftX==leftX && newLeftY==leftY) && newLeftX != 0 && newLeftY != yBorder);
					
				// отсекаем слишком короткие дуги
				if(new_arc.size() <= 16)
					continue;
				// отсекаем слишком вытянутые (смахивающие на вертикальные или горизонтальные прямые)
				double width = rightX - leftX;
				double height = abs(rightY - leftY);
				if(width/height >= lineSimilarityThreshold || height/width >= lineSimilarityThreshold)
					continue;

				// если площадь под кривой отличается от площади над кривой меньше,
				// чем на eps * totalSquare, то отбрасываем эту кривую как не выпуклую и не вогнутую
				// (слишком похожую на прямую)
				double eps = 0.1;
				if(rightY > leftY){ // I или III четверть
					// считаем площади под и над кривой
					int underSquare = 0, totalSquare = abs((rightX - leftX) * (rightY - leftY));
					auto i = new_arc.begin();
					int prev_x = i->x;
					i++;
					for(;i != new_arc.end(); i++){
						if(i->x != prev_x){ // если у кривой меняется только y, то избегаем считать один столбец дважды
							underSquare += rightY - i->y;
							prev_x = i->x;
						}
					}
					if(underSquare - (totalSquare - underSquare) > eps*totalSquare) // выпуклая вверх кривая - I 
						arcs[0].push_back(new_arc);
					else if(underSquare - (totalSquare - underSquare) < -eps*totalSquare)  // вупыклая вниз кривая - III
						arcs[2].push_back(new_arc);

				}
				else{ // II или IV 
					int underSquare = 0, totalSquare = abs((rightX - leftX) * (leftY - rightY));
					auto i = new_arc.begin();
					int prev_x = i->x;
					i++;
					for(;i != new_arc.end(); i++){
						if(i->x != prev_x){ // если у кривой меняется только y, то избегаем считать один столбец дважды
							underSquare += leftY - i->y;
							prev_x = i->x;
						}
					}
					if(underSquare - (totalSquare - underSquare) > eps*totalSquare) // выпуклая вверх кривая - II
						arcs[1].push_back(new_arc);
					else if(underSquare - (totalSquare - underSquare) < -eps*totalSquare) // вупыклая вниз кривая - IV
						arcs[3].push_back(new_arc);
				}
			}
		}
    }

	uchar aI_color[3] = {0, 0, 255};
	uchar aII_color[3] = {0, 255, 0};
	uchar aIII_color[3] = {255, 0, 0};
	uchar aIV_color[3] = {0, 255, 255};
	for(auto aI = arcs[0].begin(); aI != arcs[0].end(); aI++){
		drawArc(result, *aI, aI_color);
	}
	for(auto aII = arcs[1].begin(); aII != arcs[1].end(); aII++){
		drawArc(result, *aII, aII_color);
	}
	for(auto aIII = arcs[2].begin(); aIII != arcs[2].end(); aIII++){
		drawArc(result, *aIII, aIII_color);
	}
	for(auto aIV = arcs[3].begin(); aIV != arcs[3].end(); aIV++){
		drawArc(result, *aIV, aIV_color);
	}
	return result;
}

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
}

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
}