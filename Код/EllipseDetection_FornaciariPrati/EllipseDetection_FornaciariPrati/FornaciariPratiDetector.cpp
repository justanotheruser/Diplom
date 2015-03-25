#include "FornaciariPratiDetector.h"
#include <list>
#include "opencv2\highgui\highgui.hpp"
#include <iostream>

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

	void drawArc(const Arc& arc, cv::Mat& canvas, uchar* color){
		for(auto point : arc){
			canvas.at<cv::Vec3b>(point)[0] = color[0];
			canvas.at<cv::Vec3b>(point)[1] = color[1];
			canvas.at<cv::Vec3b>(point)[2] = color[2];
		}
	}
}

FornaciariPratiDetector::FornaciariPratiDetector(string configFile)
	: SIMILARITY_WITH_LINE_THRESHOLD(0), MINIMUM_ABOVE_UNDER_AREA_DIFFERENCE_RATIO(0)
{
}

FornaciariPratiDetector::FornaciariPratiDetector(double similarityWithLineThreshold, double minimumAboveUnderAreaDifferenceRatio)
	: SIMILARITY_WITH_LINE_THRESHOLD(similarityWithLineThreshold)
	, MINIMUM_ABOVE_UNDER_AREA_DIFFERENCE_RATIO(minimumAboveUnderAreaDifferenceRatio)
{
}

vector<Ellipse> FornaciariPratiDetector::DetectEllipses(const Mat& src)
{
	getSobelDerivatives(src);
	useCannyDetector(); // TODO: реализовать детектор Кенни самому с переиспользованием посчитанных собелей
	heuristicSearchOfArcs();
	choosePossibleTriplets();
	waitKey(0);
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
	displayImage("Canny", m_canny);
}

int FornaciariPratiDetector::calculateSquareUnderArc(const Arc& arc) const
{
	int underSquare = 0, bottomY = arc.back().y;
	auto i = arc.begin();
	int prev_x = i->x;
	i++;
	for(;i != arc.end(); i++)
	{
		if(i->x != prev_x) // если у кривой меняется только y, то избегаем считать один столбец дважды
		{
			underSquare += bottomY - i->y;
			prev_x = i->x;
		}
	}
	return underSquare;
}

void FornaciariPratiDetector::heuristicSearchOfArcs()
{
	Mat result = Mat::zeros(m_canny.rows, m_canny.cols, CV_8UC3);
	// выделяем память заранее, чтобы избежать копирования в рантайме
	for (int i = 0; i < 4; i++)
	{
		m_arcsInCoordinateQuarters[i].clear();
		m_arcsInCoordinateQuarters[i].reserve(50);
	}

	int reservedArcSize = m_canny.cols * m_canny.rows / 2 ;
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
				Arc newArc;
				newArc.reserve(reservedArcSize);
				Point upperRight, bottomLeft; // границы бокса, внутри которого наша дуга
				upperRight.y = row;
				// определяем направление градиента
				if(*sX > 0 && *sY > 0 || *sX < 0 && *sY < 0) // II или IV
				{
					findArcThatIncludesPoint(col, row, -1, newArc);
					upperRight.x = col;
					bottomLeft = newArc.back();
					if (newArc.size() <= 16) // отсекаем слишком маленькие дуги
						continue;
					// отсекаем слишком вытянутые (смахивающие на вертикальные или горизонтальные прямые)
					double width  = upperRight.x - bottomLeft.x;
					double height = bottomLeft.y - upperRight.y;
					if(width/height >= SIMILARITY_WITH_LINE_THRESHOLD || height/width >= SIMILARITY_WITH_LINE_THRESHOLD)
						continue;
					// если площадь под кривой отличается от площади над кривой меньше,
					// чем на totalSquare * MINIMUM_ABOVE_UNDER_AREA_DIFFERENCE_RATIO, то отбрасываем эту кривую как 
					// не выпуклую и не вогнутую (слишком похожую на прямую)
					int totalSquare = width * height, underSquare = calculateSquareUnderArc(newArc);
					int underMinusAbove = underSquare - (totalSquare - underSquare);
					// выпуклая вверх кривая - II
					if(underMinusAbove > MINIMUM_ABOVE_UNDER_AREA_DIFFERENCE_RATIO * totalSquare) 
						m_arcsInCoordinateQuarters[1].push_back(newArc);
					// вупыклая вниз кривая - IV
					else if(underMinusAbove < - MINIMUM_ABOVE_UNDER_AREA_DIFFERENCE_RATIO * totalSquare) 
						m_arcsInCoordinateQuarters[3].push_back(newArc);
				}
				else // I или III
				{
					findArcThatIncludesPoint(col, row, 1, newArc);
					upperRight.x = newArc.back().x;
					bottomLeft.x = col; bottomLeft.y = newArc.back().y;
					if (newArc.size() <= 16) // отсекаем слишком маленькие дуги
						continue;
					// отсекаем слишком вытянутые (смахивающие на вертикальные или горизонтальные прямые)
					double width  = upperRight.x - bottomLeft.x;
					double height = bottomLeft.y - upperRight.y;
					if(width/height >= SIMILARITY_WITH_LINE_THRESHOLD || height/width >= SIMILARITY_WITH_LINE_THRESHOLD)
						continue;
					// если площадь под кривой отличается от площади над кривой меньше,
					// чем на totalSquare * MINIMUM_ABOVE_UNDER_AREA_DIFFERENCE_RATIO, то отбрасываем эту кривую как 
					// не выпуклую и не вогнутую (слишком похожую на прямую)
					int totalSquare = width * height, underSquare = calculateSquareUnderArc(newArc);
					int underMinusAbove = underSquare - (totalSquare - underSquare);
					// выпуклая вверх кривая - I 
					if(underMinusAbove > MINIMUM_ABOVE_UNDER_AREA_DIFFERENCE_RATIO * totalSquare) 
						m_arcsInCoordinateQuarters[0].push_back(newArc);
					// вупыклая вниз кривая - III
					else if(underMinusAbove < - MINIMUM_ABOVE_UNDER_AREA_DIFFERENCE_RATIO * totalSquare) 
						m_arcsInCoordinateQuarters[2].push_back(newArc);
				}
			}
		}
	}
	uchar arcColor[4][3] = {{0, 0, 255}, {0, 255, 0}, {255, 0, 0}, {0, 255, 255}};
	for(int i = 0; i < 4; i++)
		for (auto arc : m_arcsInCoordinateQuarters[i])
			drawArc(arc, result, arcColor[i]);
	displayImage("All arcs", result);
}

void FornaciariPratiDetector::findArcThatIncludesPoint(int x, int y, int dx, Arc& arc)
{
	arc.emplace_back(x, y);
	m_canny.at<uchar>(Point(x, y)) = 0;
	Point newPoint, lastPoint;
	do // ищем новые точки дуги cбоку-по диагонали-снизу пока не перестанем находить
	{
		newPoint = lastPoint = arc.back();
		// если упёрлись в нижнюю границу, то у этой точки нет нижней и диагональной соседей
		if (lastPoint.y + 1 <= m_canny.rows-1)
		{
			// строго вниз
			newPoint = lastPoint + Point(0, 1);
			if (isEdgePoint(newPoint))
			{
				arc.push_back(newPoint);
				m_canny.at<uchar>(newPoint) = 0;
			}
			else if (lastPoint.x - 1 >= 0)
			{
				// диагональная точка
				newPoint = lastPoint + Point(dx, 1);
				if (isEdgePoint(newPoint))
				{
					arc.push_back(newPoint);
					m_canny.at<uchar>(newPoint) = 0;
				}
				else
				{
					// строго по горизонтали
					newPoint = lastPoint + Point(dx, 0);
					if (isEdgePoint(newPoint))
					{
						arc.push_back(newPoint);
						m_canny.at<uchar>(newPoint) = 0;
					}
				}
			}
		}
		else if (lastPoint.x - 1 >= 0)
		{
			// строго по горизонтали
			newPoint = lastPoint + Point(dx, 0);
			if (isEdgePoint(newPoint))
			{
				arc.push_back(newPoint);
				m_canny.at<uchar>(newPoint) = 0;
			}
		}
	} while(lastPoint != arc.back());
}

void FornaciariPratiDetector::choosePossibleTriplets(){
	// запоминаем, т.к. накладно вызывать функции в цикле на проверке условия
	int numArcsI = m_arcsInCoordinateQuarters[0].size();
	int numArcsII = m_arcsInCoordinateQuarters[1].size();
	int numArcsIII = m_arcsInCoordinateQuarters[2].size();
	int numArcsIV = m_arcsInCoordinateQuarters[3].size();
	// выделяем память заранее, чтобы избежать копирования в рантайме
	for (int i = 0; i < 4; i++)
	{
		m_possibleIandII.clear();   m_possibleIandII.reserve(numArcsI*numArcsII);
		m_possibleIIandIII.clear(); m_possibleIIandIII.reserve(numArcsII*numArcsIII);
		m_possibleIIIandIV.clear();	m_possibleIIIandIV.reserve(numArcsIII*numArcsIV);
		m_possibleIVandI.clear();   m_possibleIVandI.reserve(numArcsIV*numArcsI);
	}
	// сначала выбираем совместные пары
	for(int aI = 0; aI < numArcsI; aI++){ 
		for(int aII = 0; aII < numArcsII; aII++){
			if(m_arcsInCoordinateQuarters[1][aII].front().x <= m_arcsInCoordinateQuarters[0][aI].back().x) // aI должна быть правее aII
				m_possibleIandII.emplace_back(aI, aII);
		}
	}
	for(int aII = 0; aII < numArcsII; aII++){
		for(int aIII = 0; aIII < numArcsIII; aIII++){
			if(m_arcsInCoordinateQuarters[2][aIII].back().y >= m_arcsInCoordinateQuarters[1][aII].back().y) // aIII должна быть под aII
				m_possibleIIandIII.emplace_back(aII, aIII);
		}
	}
	for(int aIII = 0; aIII < numArcsIII; aIII++){ 
		for(int aIV = 0; aIV < numArcsIV; aIV++){
			if(m_arcsInCoordinateQuarters[2][aIII].front().x <= m_arcsInCoordinateQuarters[3][aIV].back().x) // aIII должна быть левее aIV
				m_possibleIIIandIV.emplace_back(aIII, aIV);
		}
	}
	for(int aIV = 0; aIV < numArcsIV; aIV++){
		for(int aI = 0; aI < m_arcsInCoordinateQuarters[0].size(); aI++){
			if(m_arcsInCoordinateQuarters[3][aIV].front().y >= m_arcsInCoordinateQuarters[0][aI].front().y) // aIV должна быть ниже aI
				m_possibleIVandI.emplace_back(aIV, aI);
		}
	}

	// теперь составляем возможные тройки
	int numOfPairsIandII   = m_possibleIandII.size();
	int numOfPairsIIandIII = m_possibleIIandIII.size();
	int numOfPairsIIIandIV = m_possibleIIIandIV.size();
	int numOfPairsIVandI   = m_possibleIVandI.size();
	for (int i = 0; i < 4; i++)
	{
		m_tripletsWithout_I.clear();   m_tripletsWithout_I.reserve(m_possibleIIandIII.size() * m_possibleIIIandIV.size());
		m_tripletsWithout_II.clear();  m_tripletsWithout_I.reserve(m_possibleIIIandIV.size() * m_possibleIVandI.size());
		m_tripletsWithout_III.clear(); m_tripletsWithout_I.reserve(m_possibleIandII.size() * m_possibleIVandI.size());
		m_tripletsWithout_IV.clear();  m_tripletsWithout_I.reserve(m_possibleIandII.size() * m_possibleIIandIII.size());
	}
	for(int i = 0; i < numOfPairsIandII; i++)
	{
		for(int j = 0; j < numOfPairsIIandIII; j++)
		{
			// если сегмент из второй четверти у этих пар один и тот же, то это возможная тройка
			if(m_possibleIandII[i].second == m_possibleIIandIII[j].first)
				m_tripletsWithout_IV.emplace_back(m_possibleIandII[i].first, m_possibleIandII[i].second, m_possibleIIandIII[j].second);
		}
		for(int j = 0; j < numOfPairsIVandI; j++)
		{
			// если сегмент из первой четверти у этих пар один и тот же, то это возможная тройка
			if(m_possibleIandII[i].first == m_possibleIVandI[j].second)
				m_tripletsWithout_III.emplace_back(m_possibleIandII[i].first, m_possibleIandII[i].second, m_possibleIVandI[j].first);
		}
	}
	for(int i = 0; i < numOfPairsIIIandIV; i++)
	{
		for(int j = 0; j < numOfPairsIVandI; j++)
		{
			// если сегмент из четвёртой четверти у этих пар один и тот же, то это возможная тройка
			if(m_possibleIIIandIV[i].second == m_possibleIVandI[j].first)
				m_tripletsWithout_II.emplace_back(m_possibleIVandI[i].second, m_possibleIIIandIV[i].first, m_possibleIIIandIV[j].second);
		}
		for(int j = 0; j < numOfPairsIIandIII; j++)
		{
			// если сегмент из третьей четверти у этих пар один и тот же, то это возможная тройка
			if(m_possibleIIIandIV[i].first == m_possibleIIandIII[j].second)
				m_tripletsWithout_I.emplace_back(m_possibleIandII[i].first, m_possibleIandII[i].second, m_possibleIVandI[j].first);
		}
	}	

	// without optimization
	std::cout << m_arcsInCoordinateQuarters[0].size() * m_arcsInCoordinateQuarters[1].size() * m_arcsInCoordinateQuarters[2].size() + 
				 m_arcsInCoordinateQuarters[1].size() * m_arcsInCoordinateQuarters[2].size() * m_arcsInCoordinateQuarters[3].size() +
				 m_arcsInCoordinateQuarters[2].size() * m_arcsInCoordinateQuarters[3].size() * m_arcsInCoordinateQuarters[0].size() + 
				 m_arcsInCoordinateQuarters[3].size() * m_arcsInCoordinateQuarters[0].size() * m_arcsInCoordinateQuarters[1].size() 
			  << std::endl;
	std::cout << m_tripletsWithout_I.size() + m_tripletsWithout_II.size() + 
				 m_tripletsWithout_III.size() + m_tripletsWithout_IV.size() << std::endl;
}

inline bool FornaciariPratiDetector::isEdgePoint(const Point& point)
{
	uchar* volume = m_canny.ptr(point.y);
	volume += point.x;
	return *volume == 255;
}


// TODO: подумать над тем, чтобы отслеживать среднюю точку ещё на этапе построения дуги
void FornaciariPratiDetector::findMidPoints(){
/*	m_arcsMidPoints[0].reserve(m_arcsInCoordinateQuarters[0].size());
	m_arcsMidPoints[1].reserve(m_arcsInCoordinateQuarters[1].size());
	m_arcsMidPoints[2].reserve(m_arcsInCoordinateQuarters[2].size());
	m_arcsMidPoints[3].reserve(m_arcsInCoordinateQuarters[3].size());
	for(int arcType = 0; arcType < 4; arcType++){ // цикл по типам кривых
		for (int i = 0; i < m_arcsInCoordinateQuarters[arcType].size(); i++){ // цикл по кривым одного типа
			int index = 0;
			int middle_index = m_arcsInCoordinateQuarters[arcType][i].size() / 2;
			for(auto point = arcs[arcType][i].rbegin(); point != arcs[arcType][i].rend(); point++, number++){ // цикл по точкам кривой
				if(index == middle_index){
					arcsMidPoints[arcType].push_back(point);
					break;
				}
			}
		}
	}*/
}