#include "FornaciariPratiDetector.h"
#include <list>
#include "opencv2\highgui\highgui.hpp"
#include "Arc.h"

using namespace cv;
using std::string;
using std::list;

namespace
{
	void displayImage(const char* title, const Mat& img, bool wait=false)
	{
		namedWindow(title, CV_WINDOW_AUTOSIZE);
		imshow(title, img);
		if(wait) waitKey(0);
	}
}

vector<Ellipse> FornaciariPratiDetector::DetectEllipses(const Mat& src)
{
	vector<Ellipse> ellipses;
	return ellipses;
}

vector<Ellipse> FornaciariPratiDetector::DetailedEllipseDetection(const Mat& src)
{
	// исходное изображение
	//displayImage("Source", src);

	// TODO: считать операторы Собеля эффективно
	// http://www.youtube.com/watch?v=lC-IrZsdTrw
	Mat canny, sobelX, sobelY;

	getSobelDerivatives(src);
	// TODO: реализовать детектор Кенни самому с переиспользованием посчитанных собелей
	useCannyDetector();

	// TODO: добавить альтернативный способ поиска арок:
	// InTech-Methods_for_ellipse_detection_from_edge_maps_of_real_images.pdf
	heuristicSearchOfArcs();


	/*Mat arcs = findArcs(src);
	namedWindow("Arcs", CV_WINDOW_AUTOSIZE);
	imshow("Arcs", arcs);*/


	waitKey(0);
	vector<Ellipse> ellipses;
	return ellipses;
}

void FornaciariPratiDetector::getSobelDerivatives(const Mat& src)
{
	int blurKernelSize = 3, sobelKernelSize = 3;
	double blurSigma = 1;
	GaussianBlur(src, m_blurred, Size(blurKernelSize, blurKernelSize), blurSigma, blurSigma);
	Sobel(m_blurred, m_sobelX, CV_16S, 1, 0, sobelKernelSize);
	Sobel(m_blurred, m_sobelY, CV_16S, 0, 1, sobelKernelSize);
	Mat cv_8u_sobelX, cv_8u_sobelY;
	convertScaleAbs(m_sobelX, cv_8u_sobelX);
	convertScaleAbs(m_sobelY, cv_8u_sobelY);
	//displayImage("SobelX", cv_8u_sobelX);
	//displayImage("SobelY", cv_8u_sobelY);
}

void FornaciariPratiDetector::useCannyDetector()
{
	int cannyLowTreshold = 50;
	double cannyHighLowtRatio = 2.5;
	Canny(m_blurred, m_canny, cannyLowTreshold, cannyLowTreshold * cannyHighLowtRatio);
	//displayImage("Canny", m_canny);
}

void FornaciariPratiDetector::heuristicSearchOfArcs()
{
	Arcs arcsInCoordinateQuarters[4];
	Arcs allArcs;
	for(int row = 0; row < m_canny.rows; row++)
	{ 
		uchar* p = m_canny.ptr(row);
		short* sX = m_sobelX.ptr<short>(row);
		short* sY = m_sobelY.ptr<short>(row);
		for(int col = 0; col < m_canny.cols; col++, p++, sX++, sY++) 
		{
			// start searching only from points with diagonal gradient
			if(*p == 255 && *sX != 0 && *sY != 0)
			{
				Arc newArc = findArcThatIncludesPoint(col, row, sX, sY);
				if (newArc.Size() <= 16) // отсекаем слишком маленькие дуги
					continue;
				allArcs.push_back(newArc);
			}
		}
	}

	Mat result = Mat::zeros(m_canny.rows, m_canny.cols, CV_8UC3);
	
	for (auto arc : allArcs)
	{
		uchar arcColor[3] = {rand() % 255, rand() % 255, rand() % 255};
		arc.DrawArc(result, arcColor);
	}
	displayImage("All arcs", result); 
}

Arc FornaciariPratiDetector::findArcThatIncludesPoint(int col, int row, short* sX, short* sY)
{
	Arc arc(Point(col, row));
	m_canny.at<uchar>(Point(col, row)) = 0;
	Point newPoint, lastMostRight, lastMostLeft;
	const int HORIZONTAL_LINE_THESHOLD = 20;
	const int VERTICAL_LINE_THESHOLD = 20;
	const int DIAGONAL_LINE_THESHOLD = 20;
	int verticalLineCounter = 0, horizontalLineCounter = 0, diagonalLineCounter = 0;
	// определяем направление градиента (отделяем II и IV от I и III)
	if(*sX > 0 && *sY > 0 || *sX < 0 && *sY < 0)
	{
		// ищем новые точки дуги слева пока не перестанем находить
		do
		{
			newPoint = lastMostLeft = arc.GetMostLeftPoint();
			// если упёрлись в нижнюю границу, то у этой точки нет нижней и диагональной соседей
			if (lastMostLeft.y + 1 <= m_canny.rows-1)
			{
				// строго вниз
				newPoint = lastMostLeft + Point(0, 1);
				if (isEdgePoint(newPoint) && verticalLineCounter < VERTICAL_LINE_THESHOLD)
				{
					arc.AddToTheLeft(newPoint);
					verticalLineCounter++;
					diagonalLineCounter = horizontalLineCounter = 0;
				}
				else if (lastMostLeft.x - 1 >= 0)
				{
					// диагональная точка
					newPoint = lastMostLeft + Point(-1, 1);
					if (isEdgePoint(newPoint) && diagonalLineCounter < DIAGONAL_LINE_THESHOLD)
					{
						arc.AddToTheLeft(newPoint);
						diagonalLineCounter++;
						horizontalLineCounter = verticalLineCounter = 0;
					}
					else
					{
						// строго влево
						newPoint = lastMostLeft + Point(-1, 0);
						if (isEdgePoint(newPoint) && horizontalLineCounter < HORIZONTAL_LINE_THESHOLD)
						{
							arc.AddToTheLeft(newPoint);
							horizontalLineCounter++;
							diagonalLineCounter = verticalLineCounter = 0;
						}
					}
				}
			}
			else if (lastMostLeft.x - 1 >= 0)
			{
				// строго влево
				newPoint = lastMostLeft + Point(-1, 0);
				if (isEdgePoint(newPoint) && horizontalLineCounter < HORIZONTAL_LINE_THESHOLD)
				{
					arc.AddToTheLeft(newPoint);
					horizontalLineCounter++;
					diagonalLineCounter = verticalLineCounter = 0;
				}
			}
		} while(lastMostLeft != arc.GetMostLeftPoint());
	}
	else
	{
		do
		{
			newPoint = lastMostRight = arc.GetMostRightPoint();
			// если упёрлись в нижнюю границу, то у этой точки нет нижней и диагональной соседей
			if (lastMostRight.y + 1 <= m_canny.rows-1)
			{
				// строго вниз
				newPoint = lastMostRight + Point(0, 1);
				if (isEdgePoint(newPoint) && verticalLineCounter < VERTICAL_LINE_THESHOLD)
				{
					arc.AddToTheRight(newPoint);
					verticalLineCounter++;
					diagonalLineCounter = horizontalLineCounter = 0;

				}
				else if (lastMostRight.x + 1 <= m_canny.cols-1)
				{
					// диагональная точка
					newPoint = lastMostRight + Point(1, 1);
					if (isEdgePoint(newPoint) && diagonalLineCounter < DIAGONAL_LINE_THESHOLD)
					{
						arc.AddToTheRight(newPoint);
						diagonalLineCounter++;
						verticalLineCounter = horizontalLineCounter = 0;

					}
					else
					{
						// строго влево
						newPoint = lastMostRight + Point(1, 0);
						if (isEdgePoint(newPoint) && horizontalLineCounter < HORIZONTAL_LINE_THESHOLD)
						{
							arc.AddToTheRight(newPoint);
							horizontalLineCounter++;
							diagonalLineCounter = verticalLineCounter = 0;

						}
					}
				}
			}
			else if (lastMostRight.x + 1 <= m_canny.cols-1)
			{
				// строго влево
				newPoint = lastMostRight + Point(1, 0);
				if (isEdgePoint(newPoint) && horizontalLineCounter < HORIZONTAL_LINE_THESHOLD)
				{
					arc.AddToTheRight(newPoint);
					horizontalLineCounter++;
					diagonalLineCounter = verticalLineCounter = 0;
				}
			}
		} while(lastMostRight != arc.GetMostRightPoint());
	}
	return arc;
}

Mat FornaciariPratiDetector::findArcs(const Mat& src){
	vector<list<Point>> arcs[4];
	Mat result = Mat::zeros(m_canny.rows, m_canny.cols, CV_8UC3);

	// используется в двух местах для отсекания кривых, смахивающих на линии
	/*const int lineSimilarityThreshold = 20; // предельное число точек подряд, у которых одна из координат неизменна

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
	}*/
	return result;
}


bool FornaciariPratiDetector::isEdgePoint(const Point& point)
{
	uchar* volume = m_canny.ptr(point.y);
	volume += point.x;
	if (*volume == 255){
		*volume = 0;
		return true;
	}
	return false;
}