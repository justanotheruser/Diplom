#include "FornaciariPratiDetector.h"
#include "HoughTransformAccumulator.h"
#include "opencv2\highgui\highgui.hpp"
#define _USE_MATH_DEFINES
#include <math.h>
#include <tuple>

using namespace cv;
using std::string;
using std::list;

uchar arcColor[4][3] = {{0, 0, 255}, {0, 255, 0}, {255, 0, 0}, {0, 255, 255}};
const int sc_maxDiscrepancyForParallels = 150;
const int sc_doubledNumberOfChords = 10;

double getSlope(const std::vector<Point>& midPoints)
{
	int middle = midPoints.size() / 2;
	double slope = 0;
	unsigned int i;
	for (i = 0; i + middle < midPoints.size(); i++)
	{
	   double xDiff = midPoints[i].x - midPoints[i + middle].x;
	   double yDiff = midPoints[i].y - midPoints[i + middle].y;
	   slope +=  yDiff / xDiff;
	}
	slope /= i;
	return slope;
}

Point getAveragePoint(const std::vector<Point>& midPoints)
{
	Point avrPoint;
	for (auto point : midPoints)
	{
		avrPoint += point;
	}
	avrPoint.x /= midPoints.size();
	avrPoint.y /= midPoints.size();
	return avrPoint;
}

void displayImage(const char* title, const Mat& img, bool wait=false)
{
	namedWindow(title, CV_WINDOW_AUTOSIZE);
	imshow(title, img);
	//imwrite("../.."+std::string(title)+".jpg", img);
	if(wait) waitKey(0);
}

void drawArc(const Arc& arc, cv::Mat& canvas, uchar* color){
		for(auto point : arc){
			canvas.at<cv::Vec3b>(point)[0] = color[0];
			canvas.at<cv::Vec3b>(point)[1] = color[1];
			canvas.at<cv::Vec3b>(point)[2] = color[2];
		}
	}

bool curvatureCondition(const Arc& firstArc, const Arc& secondArc)
{
	Point firstMidPoint = firstArc[firstArc.size()/2];
	Point secondMidPoint = secondArc[secondArc.size()/2];
	Point firstC = firstArc.back() + firstArc.front();
	firstC.x /= 2;
	firstC.y /= 2;
	Point secondC = secondArc.back() + secondArc.front();
	secondC.x /= 2;
	secondC.y /= 2;
	double midPointsDist = cv::norm(firstMidPoint-secondMidPoint);
	if (midPointsDist > cv::norm(firstC-secondMidPoint) && 
		midPointsDist > cv::norm(secondC-firstMidPoint))
		return true;
	return false;
}

double findMajorAxis(const Arc& arc_1, const Arc& arc_2, const Arc& arc_3, const Point& center, double N, double K)
{
	double x0, y0;
	HoughTransformAccumulator acc;
	for (auto point : arc_1)
	{
		x0 = (point.x - center.x + (point.y - center.y)*K) / sqrt(K*K + 1);
		y0 = (-(point.x - center.x)*K + point.y - center.y) / sqrt(K*K + 1);
		double newAx = std::sqrt((x0*x0*N*N + y0*y0) / (N*N*(K*K + 1)));
		// округление до тысячных
		double rounded = floor(1000*newAx + 0.5) / 1000.0;
		acc.Add(rounded);
	}
	for (auto point : arc_2)
	{
		x0 = (point.x - center.x + (point.y - center.y)*K) / sqrt(K*K + 1);
		y0 = (-(point.x - center.x)*K + point.y - center.y) / sqrt(K*K + 1);
		double newAx = std::sqrt((x0*x0*N*N + y0*y0) / (N*N*(K*K + 1)));
		double rounded = floor(1000*newAx + 0.5) / 1000.0;
		acc.Add(rounded);
	}
	for (auto point : arc_3)
	{
		x0 = (point.x - center.x + (point.y - center.y)*K) / sqrt(K*K + 1);
		y0 = (-(point.x - center.x)*K + point.y - center.y) / sqrt(K*K + 1);
		double newAx = std::sqrt((x0*x0*N*N + y0*y0) / (N*N*(K*K + 1)));
		double rounded = floor(1000*newAx + 0.5) / 1000.0;
		acc.Add(rounded);
	}
	double Ax = acc.FindMax();
	double Ay = abs(Ax * K);
	return std::sqrt(Ax*Ax + Ay*Ay);
}

// находят уравнение прямой, проходящей через середины параллельных хорд

std::tuple<double, int> findLineCrossingMidpointBetweenHorizontalArcs(const Arc& arcI, const Arc& arcII)
{
	std::vector<Point> midPoints;
	// a) построить прямую между нижним краем дуги I и серединой дуги II
	Point lowestPoint = arcI[arcI.size()-1];
	Point midPoint = arcII[arcII.size()/2];
	//line(chords, lowestPoint, midPoint, Scalar(200, 200, 200));
	Point dir_vector = lowestPoint - midPoint; // направляющий вектор исходной хорды
	midPoints.emplace_back((lowestPoint.x + midPoint.x) / 2, (lowestPoint.y + midPoint.y) / 2);
	// б) построить ещё несколько параллельных ей
	int delta = arcI.size() / sc_doubledNumberOfChords;
	if (delta <= 0) delta = 1;
	// каждая последующая хорда будет пересекать дугу I выше и выше, поэтому запоминаем, откуда нам 
	// следует искать следующую точку
	int checkpoint =  arcI.size() - 2;
	
	for (int pointII = arcII.size()/2 - delta; pointII >= 0; pointII -= delta)
	{
		// ищем точку на дуге I, которую пересекает хорда
		// будем брать ту, которая после подстановки в уравнение прямой с данным коэфеициентом С и
		// данным направляющим вектором даёт наименьшую невязку (отклонение от 0).
		// коэффициент в уравнении прямой, проходящей через текущую точку первой дуги
		int С = dir_vector.x * arcII[pointII].y - dir_vector.y * arcII[pointII].x; 
		int prev_discrepancy = abs(dir_vector.y * arcI[checkpoint].x - dir_vector.x * arcI[checkpoint].y + С);
		checkpoint--;
		bool haventFoundChord = false;
		for (int pointI = checkpoint; pointI > 0; pointI--)
		{
			int discrepancy = abs(dir_vector.y * arcI[pointI].x - dir_vector.x * arcI[pointI].y + С);
			if (discrepancy > prev_discrepancy)
			{
				// дополнительная проверка на минимум (иногда невязка продолжает падать со следующей точки)
				pointI--;
				int next_discrepancy = abs(dir_vector.y * arcI[pointI].x - dir_vector.x * arcI[pointI].y + С);
				if (next_discrepancy < prev_discrepancy)
				{
					// продолжаем поиск минимума
					prev_discrepancy = next_discrepancy;
					continue;
				}
				else if (prev_discrepancy <= sc_maxDiscrepancyForParallels)
				{
					// мы нашли минимум, и он нас устраивает
					checkpoint = pointI+2;
					break;
				}
				else
				{
					// нас не устраивает наилучший вариант из тех, что мы нашли
					// переходим к поиску параллельной хорды от следующей точки дуги I
					haventFoundChord = true;
					break;
				}
			}
			prev_discrepancy = discrepancy;
		}
		if (haventFoundChord)
			continue;
		// в) найди середины хорд
		midPoints.emplace_back((arcII[pointII].x + arcI[checkpoint].x) / 2, 
							   (arcII[pointII].y + arcI[checkpoint].y) / 2);
	}
	double slope = getSlope(midPoints);
	Point pointOnLineCrossingCenter = getAveragePoint(midPoints);
	// y = slope*x + coeff
	int coeff = pointOnLineCrossingCenter.y - slope * pointOnLineCrossingCenter.x;
	return std::make_tuple(slope, coeff);
}

std::tuple<double, int> findLineCrossingMidpointBetweenUpperAndLowerArcs(const Arc& arcI, const Arc& arcIV)
{
	std::vector<Point> midPoints;
	// a) построить прямую между верхним краем дуги I и серединой дуги IV
	Point highestPoint = arcI[0];
	Point midPoint = arcIV[arcIV.size()/2];
	//line(chords, highestPoint, midPoint, Scalar(200, 200, 200));
	Point dir_vector = highestPoint - midPoint; // направляющий вектор исходной хорды
	midPoints.emplace_back((highestPoint.x + midPoint.x) / 2, (highestPoint.y + midPoint.y) / 2);
	// б) построить ещё несколько параллельных ей
	int delta = arcIV.size() / sc_doubledNumberOfChords;
	if (delta <= 0) delta = 1;
	// каждая последующая хорда будет пересекать дугу I ниже и ниже, поэтому запоминаем, откуда нам 
	// следует искать следующую точку
	int checkpoint = 0;
	
	for (int pointIV = arcIV.size()/2 - delta; pointIV >= delta; pointIV -= delta)
	{
		// ищем точку на дуге I, которую пересекает хорда
		// будем брать ту, которая после подстановки в уравнение прямой с данным коэфеициентом С и
		// данным направляющим вектором даёт наименьшую невязку (отклонение от 0).
		// коэффициент в уравнении прямой, проходящей через текущую точку первой дуги
		int С = dir_vector.x * arcIV[pointIV].y - dir_vector.y * arcIV[pointIV].x; 
		int prev_discrepancy = abs(dir_vector.y * arcI[checkpoint].x - dir_vector.x * arcI[checkpoint].y + С);
		checkpoint++;
		bool haventFoundChord = false;
		for (int pointI = checkpoint; pointI < arcI.size(); pointI++)
		{
			int discrepancy = abs(dir_vector.y * arcI[pointI].x - dir_vector.x * arcI[pointI].y + С);
			if (discrepancy > prev_discrepancy)
			{
				// дополнительная проверка на минимум (иногда невязка продолжает падать со следующей точки)
				pointI++;
				int next_discrepancy = abs(dir_vector.y * arcI[pointI].x - dir_vector.x * arcI[pointI].y + С);
				if (next_discrepancy < prev_discrepancy)
				{
					// продолжаем поиск минимума
					prev_discrepancy = next_discrepancy;
					continue;
				}
				else if (prev_discrepancy <= sc_maxDiscrepancyForParallels)
				{
					// мы нашли минимум, и он нас устраивает
					checkpoint = pointI-2;
					break;
				}
				else
				{
					// нас не устраивает наилучший вариант из тех, что мы нашли
					// переходим к поиску параллельной хорды от следующей точки дуги I
					haventFoundChord = true;
					break;
				}
			}
			prev_discrepancy = discrepancy;
		}
		if (haventFoundChord)
			continue;
		// в) найди середины хорд
		midPoints.emplace_back((arcIV[pointIV].x + arcI[checkpoint].x) / 2, 
							   (arcIV[pointIV].y + arcI[checkpoint].y) / 2);
	}
	double slope = getSlope(midPoints);
	Point pointOnLineCrossingCenter = getAveragePoint(midPoints);
	// y = slope*x + coeff
	int coeff = pointOnLineCrossingCenter.y - slope * pointOnLineCrossingCenter.x;
	return std::make_tuple(slope, coeff);
}

std::tuple<double, int> findLineCrossingMidpointBetweenLowerAndUpperArcs(const Arc& arcIV, const Arc& arcI)
{
	std::vector<Point> midPoints;
	// a) построить прямую между нижним краем второй дуги и серединой первой
	Point lowestPoint = arcIV[arcIV.size()-1];
	Point midPoint = arcI[arcI.size()/2];
	Point dir_vector = lowestPoint - midPoint; // направляющий вектор исходной хорды
	midPoints.emplace_back((lowestPoint.x + midPoint.x) / 2, (lowestPoint.y + midPoint.y) / 2);
	// б) построить ещё несколько параллельных ей
	// последняя должна иметь крайней точкой нижний край дуги I
	int delta = arcI.size() / sc_doubledNumberOfChords;
	if (delta <= 0) delta = 1;
	// каждая последующая хорда будет пересекать дугу IV выше и выше, поэтому запоминаем, откуда нам 
	// следует искать следующую точку
	int checkpoint = arcIV.size()-1;
	
	for (int pointI = arcI.size()/2 + delta; pointI < arcI.size() - delta; pointI += delta)
	{
		// ищем точку на дуге IV, которую пересекает хорда
		// будем брать ту, которая после подстановки в уравнение прямой с данным коэфеициентом С и
		// данным направляющим вектором даёт наименьшую невязку (отклонение от 0).
		// коэффициент в уравнении прямой, проходящей через текущую точку первой дуги
		int С = dir_vector.x * arcI[pointI].y - dir_vector.y * arcI[pointI].x; 
		int prev_discrepancy = abs(dir_vector.y * arcIV[checkpoint].x - dir_vector.x * arcIV[checkpoint].y + С);
		checkpoint--;
		bool haventFoundChord = false;
		for (int pointIV = checkpoint; pointIV > 0; pointIV--)
		{
			int discrepancy = abs(dir_vector.y * arcIV[pointIV].x - dir_vector.x * arcIV[pointIV].y + С);
			if (discrepancy > prev_discrepancy)
			{
				// дополнительная проверка на минимум (иногда невязка продолжает падать со следующей точки)
				pointIV--;
				int next_discrepancy = abs(dir_vector.y * arcIV[pointIV].x - dir_vector.x * arcIV[pointIV].y + С);
				if (next_discrepancy < prev_discrepancy)
				{
					// продолжаем поиск минимума
					prev_discrepancy = next_discrepancy;
					continue;
				}
				else if (prev_discrepancy <= sc_maxDiscrepancyForParallels)
				{
					// мы нашли минимум, и он нас устраивает
					checkpoint = pointIV+2;
					break;
				}
				else
				{
					// нас не устраивает наилучший вариант из тех, что мы нашли
					// переходим к поиску параллельной хорды от следующей точки дуги I
					haventFoundChord = true;
					break;
				}
			}
			prev_discrepancy = discrepancy;
		}
		if (haventFoundChord)
			continue;
		// в) найди середины хорд
		midPoints.emplace_back((arcI[pointI].x + arcIV[checkpoint].x) / 2, 
							   (arcI[pointI].y + arcIV[checkpoint].y) / 2);
	}
	double slope = getSlope(midPoints);
	Point pointOnLineCrossingCenter = getAveragePoint(midPoints);
	// y = slope*x + coeff
	int coeff = pointOnLineCrossingCenter.y - slope * pointOnLineCrossingCenter.x;
	return std::make_tuple(slope, coeff);
}

cv::Point findIntersection(std::tuple<double /*slope*/, int /*coeff*/> line_1, std::tuple<double, int> line_2)
{
	// y = x*slope_1 + coeff_1
	// y = x*slope_2 + coeff_2

	// x*slope_2 + coeff_2 = x*slope_1 + coeff_1
	// x*(slope_2 - slope_1) = coeff_1 - coeff_2
	// x = (coeff_1 - coeff_2) / (slope_2 - slope_1)
	int x = static_cast<int>((std::get<1>(line_1) - std::get<1>(line_2)) 
		/ (std::get<0>(line_2) - std::get<0>(line_1)));
	int y = x*std::get<0>(line_1) + std::get<1>(line_1);
	return Point(x, y);
}

std::tuple<double, double, double> getEllipseParams(double q1, double q2, double q3, double q4)
{
	double alpha = q1*q2 - q3*q4;
	double beta = (q3*q4 + 1)*(q1 + q2) - (q1*q2 + 1)*(q3 + q4);
	double K_plus =	(-beta + std::sqrt(beta*beta + 4*alpha*alpha)) / (2*alpha);
	double N_plus = std::sqrt(-(q1 - K_plus) * (q2 - K_plus) / ((1 + q1*K_plus)*(1 + q2*K_plus)));
	double N, K, ro;
	
	if (N_plus <= 1)
	{
		N = N_plus;
		ro = std::atan(K_plus);
		K = K_plus;
	}
	else
	{
		N = 1/N_plus;
		ro = std::atan(K_plus) + M_PI/2;
		K = std::tan(ro);
	}
	std::cout << "N_plus " << N_plus << std::endl;
	std::cout << "N " << N << std::endl;
	std::cout << "K_plus " << K_plus << std::endl;
	std::cout << "K " << K << std::endl;
	std::cout << "ro " << ro << std::endl;
	return std::tuple<double, double, double>(N, K, ro);
}

FornaciariPratiDetector::FornaciariPratiDetector(string configFile)
	: SIMILARITY_WITH_LINE_THRESHOLD(0), MINIMUM_ABOVE_UNDER_AREA_DIFFERENCE_RATIO(0)
{
}

FornaciariPratiDetector::FornaciariPratiDetector(double similarityWithLineThreshold, 
												 double minimumAboveUnderAreaDifferenceRatio,
												 int blurKernelSize, int blurSigma, int sobelKernelSize)

	: SIMILARITY_WITH_LINE_THRESHOLD(similarityWithLineThreshold)
	, MINIMUM_ABOVE_UNDER_AREA_DIFFERENCE_RATIO(minimumAboveUnderAreaDifferenceRatio)
	, m_blurKernelSize(blurKernelSize)
	, m_blurSigma(blurSigma)
	, m_sobelKernelSize(sobelKernelSize)

{
}

vector<Ellipse> FornaciariPratiDetector::DetectEllipses(const Mat& src)
{
	getSobelDerivatives(src);
	useCannyDetector(); // TODO: реализовать детектор Кенни самому с переиспользованием посчитанных собелей
	heuristicSearchOfArcs();
	choosePossibleTriplets();
	testTriplets();
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
	GaussianBlur(src, m_blurred, Size(m_blurKernelSize, m_blurKernelSize), m_blurSigma, m_blurSigma);
	Sobel(m_blurred, m_sobelX, CV_16S, 1, 0, m_sobelKernelSize);
	Sobel(m_blurred, m_sobelY, CV_16S, 0, 1, m_sobelKernelSize);
#ifdef _DEBUG 
	Mat cv_8u_sobelX, cv_8u_sobelY;
	convertScaleAbs(m_sobelX, cv_8u_sobelX);
	convertScaleAbs(m_sobelY, cv_8u_sobelY);
	displayImage("SobelX", cv_8u_sobelX);
	displayImage("SobelY", cv_8u_sobelY);
#endif
}

void FornaciariPratiDetector::useCannyDetector()
{
	int cannyLowTreshold = 50;
	double cannyHighLowtRatio = 2.5;
	Canny(m_blurred, m_canny, cannyLowTreshold, cannyLowTreshold * cannyHighLowtRatio);
#ifdef _DEBUG
	displayImage("Canny", m_canny);
#endif
}

void FornaciariPratiDetector::heuristicSearchOfArcs()
{
	// выделяем память заранее, чтобы избежать копирования в рантайме
	for (int i = 0; i < 4; i++)
	{
		m_arcsInCoordinateQuarters[i].clear();
		m_arcsInCoordinateQuarters[i].reserve(50);
	}

	int reservedArcSize = m_canny.cols * m_canny.rows / 2;
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
#ifdef _DEBUG
	Mat result = Mat::zeros(m_canny.rows, m_canny.cols, CV_8UC3);
	uchar arcColor[4][3] = {{0, 0, 255}, {0, 255, 0}, {255, 0, 0}, {0, 255, 255}};
	for(int i = 0; i < 4; i++)
		for (auto arc : m_arcsInCoordinateQuarters[i])
			drawArc(arc, result, arcColor[i]);
	displayImage("All arcs", result);
#endif
}

void FornaciariPratiDetector::findArcThatIncludesPoint(int x, int y, int dx, Arc& arc)
{
	const int VERTICAL_LINE_THRESHOLD = 20;
	const int HORIZONTAL_LINE_THRESHOLD = 20;
	int looksLikeHorizontalLine = 0, looksLikeVerticalLine = 0;

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
				arc.push_back(newPoint);
			else if (lastPoint.x - 1 >= 0)
			{
				// диагональная точка
				newPoint = lastPoint + Point(dx, 1);
				if (isEdgePoint(newPoint))
					arc.push_back(newPoint);
				else
				{
					// строго по горизонтали 
					newPoint = lastPoint + Point(dx, 0);
					if (isEdgePoint(newPoint))
						arc.push_back(newPoint);
				}
			}
		}
		else if (lastPoint.x - 1 >= 0)
		{
			// строго по горизонтали
			newPoint = lastPoint + Point(dx, 0);
			if (isEdgePoint(newPoint))
				arc.push_back(newPoint);
		}
	} while(lastPoint != arc.back());
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
	Mat curvatureConditionMat = Mat::zeros(m_canny.rows, m_canny.cols, CV_8UC3);
	int cond1 = 0, cond2 = 0;
	for(int aI = 0; aI < numArcsI; aI++){ 
		for(int aII = 0; aII < numArcsII; aII++)
		{
			// aI должна быть правее aII
			if(m_arcsInCoordinateQuarters[1][aII].front().x <= m_arcsInCoordinateQuarters[0][aI].front().x)
			{
				if(curvatureCondition(m_arcsInCoordinateQuarters[0][aI], m_arcsInCoordinateQuarters[1][aII]))
				{
					m_possibleIandII.emplace_back(aI, aII);
				}
			}
		}
	}
	for(int aII = 0; aII < numArcsII; aII++){
		for(int aIII = 0; aIII < numArcsIII; aIII++)
		{
			// aIII должна быть под aII
			if(m_arcsInCoordinateQuarters[2][aIII].front().y >= m_arcsInCoordinateQuarters[1][aII].back().y)
			{
				
				if(curvatureCondition(m_arcsInCoordinateQuarters[1][aII], m_arcsInCoordinateQuarters[2][aIII]))
					m_possibleIIandIII.emplace_back(aII, aIII);
			}
		}
	}
	for(int aIII = 0; aIII < numArcsIII; aIII++){ 
		for(int aIV = 0; aIV < numArcsIV; aIV++)
		{
			// aIII должна быть левее aIV
			if(m_arcsInCoordinateQuarters[2][aIII].back().x <= m_arcsInCoordinateQuarters[3][aIV].back().x)
			{
				if(curvatureCondition(m_arcsInCoordinateQuarters[2][aIII], m_arcsInCoordinateQuarters[3][aIV]))
					m_possibleIIIandIV.emplace_back(aIII, aIV);
			}
		}
	}
	for(int aIV = 0; aIV < numArcsIV; aIV++){
		for(int aI = 0; aI < m_arcsInCoordinateQuarters[0].size(); aI++)
		{
			// aIV должна быть ниже aI
			if(m_arcsInCoordinateQuarters[3][aIV].front().y >= m_arcsInCoordinateQuarters[0][aI].back().y)
				if(curvatureCondition(m_arcsInCoordinateQuarters[3][aIV], m_arcsInCoordinateQuarters[0][aI]))
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
				m_tripletsWithout_II.emplace_back(m_possibleIVandI[j].second, m_possibleIIIandIV[i].first, m_possibleIIIandIV[i].second);
		}
		for(int j = 0; j < numOfPairsIIandIII; j++)
		{
			// если сегмент из третьей четверти у этих пар один и тот же, то это возможная тройка
			if(m_possibleIIIandIV[i].first == m_possibleIIandIII[j].second)
				m_tripletsWithout_I.emplace_back(m_possibleIIandIII[j].first, m_possibleIIIandIV[i].first, m_possibleIIIandIV[j].second);
		}
	}	
	
	std::cout << "Number of possible triplets of arcs without optimization: "
			  << m_arcsInCoordinateQuarters[0].size() * m_arcsInCoordinateQuarters[1].size() * m_arcsInCoordinateQuarters[2].size() + 
				 m_arcsInCoordinateQuarters[1].size() * m_arcsInCoordinateQuarters[2].size() * m_arcsInCoordinateQuarters[3].size() +
				 m_arcsInCoordinateQuarters[2].size() * m_arcsInCoordinateQuarters[3].size() * m_arcsInCoordinateQuarters[0].size() + 
				 m_arcsInCoordinateQuarters[3].size() * m_arcsInCoordinateQuarters[0].size() * m_arcsInCoordinateQuarters[1].size() 
			  << std::endl;
	std::cout << "With it: " << m_tripletsWithout_I.size() + m_tripletsWithout_II.size() + 
				                m_tripletsWithout_III.size() + m_tripletsWithout_IV.size() 
			  << std::endl;
#ifdef _DEBUG
	for (auto triplet : m_tripletsWithout_I)
	{
		m_remainingArcsIdx[1].insert(triplet[0]);
		m_remainingArcsIdx[2].insert(triplet[1]);
		m_remainingArcsIdx[3].insert(triplet[2]);
	}
	for (auto triplet : m_tripletsWithout_II)
	{
		m_remainingArcsIdx[0].insert(triplet[0]);
		m_remainingArcsIdx[2].insert(triplet[1]);
		m_remainingArcsIdx[3].insert(triplet[2]);
	}
	for (auto triplet : m_tripletsWithout_III)
	{
		m_remainingArcsIdx[0].insert(triplet[0]);
		m_remainingArcsIdx[1].insert(triplet[1]);
		m_remainingArcsIdx[3].insert(triplet[2]);
	}
	for (auto triplet : m_tripletsWithout_IV)
	{
		m_remainingArcsIdx[0].insert(triplet[0]);
		m_remainingArcsIdx[1].insert(triplet[1]);
		m_remainingArcsIdx[2].insert(triplet[2]);
	}

	Mat result = Mat::zeros(m_canny.rows, m_canny.cols, CV_8UC3);
	for (int q = 0; q < 4; q++)
		for (int i : m_remainingArcsIdx[q])
			drawArc(m_arcsInCoordinateQuarters[q][i], result, arcColor[q]);
	displayImage("All possible triplets", result);
#endif
}

inline bool FornaciariPratiDetector::isEdgePoint(const Point& point)
{
	uchar* volume = m_canny.ptr(point.y);
	volume += point.x;
	if (*volume > 0){
		*volume = 0;
		return true;
	}
	return false;
}

void FornaciariPratiDetector::testTriplets()
{
	for(auto triplet : m_tripletsWithout_IV)
	{
		/*Mat chords = Mat::zeros(m_canny.rows, m_canny.cols, CV_8UC3);
		Arc arcI = m_arcsInCoordinateQuarters[0][triplet[0]];
		Arc arcII = m_arcsInCoordinateQuarters[1][triplet[1]];
		Arc arcIII = m_arcsInCoordinateQuarters[2][triplet[2]];
		drawArc(arcI, chords, arcColor[0]);
		drawArc(arcII, chords, arcColor[1]);
		drawArc(arcIII, chords, arcColor[2]);
		
		// ищем прямые, проходящие через середины параллельных хорд
		// они пересекаются в центре эллипса
		auto line_12 = findLineCrossingMidpointBetweenHorizontalArcs(arcI, arcII);
		auto line_21 = findLineCrossingMidpointBetweenHorizontalArcs(arcII, arcI);
		Point intersection_1 = findIntersection(line_12, line_21);
		std::cout << intersection_1 << std::endl;
		auto line_23 = findLineCrossingMidpointBetweenUpperAndLowerArcs(arcII, arcIII);
		auto line_32 = findLineCrossingMidpointBetweenLowerAndUpperArcs(arcIII, arcII);
		Point intersection_2 = findIntersection(line_23, line_32);

		double q1, q2, q3, q4;
		q1 = static_cast<double>((arcII[arcII.size()-1].y - arcI[arcI.size()/2].y)) / 
			 (arcII[arcII.size()-1].x - arcI[arcI.size()/2].x);
		q2 = std::get<0>(line_21);
		q3 = static_cast<double>((arcIII[arcIII.size()-1].y - arcII[arcII.size()/2].y)) / 
			 (arcIII[arcIII.size()-1].x - arcII[arcII.size()/2].x);
		q4 = std::get<0>(line_32);

		double N, K, ro;
		std::tie(N, K, ro) = getEllipseParams(q1, q2, q3, q4);
		 
		Point center((intersection_1.x + intersection_2.x) / 2,
					 (intersection_1.y  + intersection_2.y) / 2);
		std::cout << "Center: " << center << std::endl;
			
		int A = abs(findMajorAxis(arcI, arcII, arcIII, center, N, K));
		double B = A * N;
		double roInDegrees = ro/M_PI * 180;
		std::cout << "roInDegrees " << roInDegrees << std::endl;
		std::cout << "A " << A << " B " << B << std::endl;
		ellipse(chords, center, Size(A, B), roInDegrees, 0, 360, Scalar(0, 0, 255));
		displayImage("Chords", chords);*/
	}

	for(auto triplet : m_tripletsWithout_III)
	{
	    /*Mat chords = Mat::zeros(m_canny.rows, m_canny.cols, CV_8UC3);
		Arc arcI = m_arcsInCoordinateQuarters[0][triplet[0]];
		Arc arcII = m_arcsInCoordinateQuarters[1][triplet[1]];
		Arc arcIV = m_arcsInCoordinateQuarters[3][triplet[2]];
		drawArc(arcI, chords, arcColor[0]);
		drawArc(arcII, chords, arcColor[1]);
		drawArc(arcIV, chords, arcColor[3]);
		
		// ищем прямые, проходящие через середины параллельных хорд
		// они пересекаются в центре эллипса
		auto line_12 = findLineCrossingMidpointBetweenHorizontalArcs(arcI, arcII);
		auto line_21 = findLineCrossingMidpointBetweenHorizontalArcs(arcII, arcI);
		Point intersection_1 = findIntersection(line_12, line_21);
		std::cout << intersection_1 << std::endl;
		auto line_14 = findLineCrossingMidpointBetweenUpperAndLowerArcs(arcI, arcIV);
		auto line_41 = findLineCrossingMidpointBetweenLowerAndUpperArcs(arcIV, arcI);
		Point intersection_2 = findIntersection(line_14, line_41);

		double q1, q2, q3, q4;
		q1 = static_cast<double>((arcII[arcII.size()-1].y - arcI[arcI.size()/2].y)) / 
			 (arcII[arcII.size()-1].x - arcI[arcI.size()/2].x);
		q2 = std::get<0>(line_21);
		q3 = static_cast<double>((arcIV[arcIV.size()-1].y - arcI[arcI.size()/2].y)) / 
			 (arcIV[arcIV.size()-1].x - arcI[arcI.size()/2].x);
		q4 = std::get<0>(line_41);

		double N, K, ro;
		std::tie(N, K, ro) = getEllipseParams(q1, q2, q3, q4);
		 
		Point center((intersection_1.x + intersection_2.x) / 2,
					 (intersection_1.y  + intersection_2.y) / 2);
		std::cout << "Center: " << center << std::endl;
			
		int A = abs(findMajorAxis(arcI, arcII, arcIV, center, N, K));
		double B = A * N;
		double roInDegrees = ro/M_PI * 180;
		std::cout << "roInDegrees " << roInDegrees << std::endl;
		std::cout << "A " << A << " B " << B << std::endl;
		ellipse(chords, center, Size(A, B), roInDegrees, 0, 360, Scalar(0, 0, 255));
		displayImage("Chords", chords);*/
	}

	for(auto triplet : m_tripletsWithout_II)
	{
	    /*Mat chords = Mat::zeros(m_canny.rows, m_canny.cols, CV_8UC3);
		Arc arcI = m_arcsInCoordinateQuarters[0][triplet[0]];
		Arc arcIII = m_arcsInCoordinateQuarters[2][triplet[1]];
		Arc arcIV = m_arcsInCoordinateQuarters[3][triplet[2]];
		drawArc(arcI, chords, arcColor[0]);
		drawArc(arcIII, chords, arcColor[2]);
		drawArc(arcIV, chords, arcColor[3]);
		
		// ищем прямые, проходящие через середины параллельных хорд
		// они пересекаются в центре эллипса
		auto line_34 = findLineCrossingMidpointBetweenHorizontalArcs(arcIII, arcIV);
		auto line_43 = findLineCrossingMidpointBetweenHorizontalArcs(arcIV, arcIII);
		Point intersection_1 = findIntersection(line_34, line_43);
		std::cout << intersection_1 << std::endl;
		auto line_14 = findLineCrossingMidpointBetweenUpperAndLowerArcs(arcI, arcIV);
		auto line_41 = findLineCrossingMidpointBetweenLowerAndUpperArcs(arcIV, arcI);
		Point intersection_2 = findIntersection(line_14, line_41);

		double q1, q2, q3, q4;
		q1 = static_cast<double>((arcIV[arcIV.size()-1].y - arcIII[arcIII.size()/2].y)) / 
			 (arcIV[arcIV.size()-1].x - arcIII[arcIII.size()/2].x);
		q2 = std::get<0>(line_43);
		q3 = static_cast<double>((arcIV[arcIV.size()/2].y - arcI[0].y)) / 
			 (arcIV[arcIV.size()/2].x - arcI[0].x);
		q4 = std::get<0>(line_14);

		double N, K, ro;
		std::tie(N, K, ro) = getEllipseParams(q1, q2, q3, q4);
		 
		Point center((intersection_1.x + intersection_2.x) / 2,
					 (intersection_1.y  + intersection_2.y) / 2);
		std::cout << "Center: " << center << std::endl;
			
		int A = abs(findMajorAxis(arcI, arcIII, arcIV, center, N, K));
		double B = A * N;
		double roInDegrees = ro/M_PI * 180;
		std::cout << "roInDegrees " << roInDegrees << std::endl;
		std::cout << "A " << A << " B " << B << std::endl;
		ellipse(chords, center, Size(A, B), roInDegrees, 0, 360, Scalar(0, 0, 255));
		displayImage("Chords", chords);*/
	}

	for(auto triplet : m_tripletsWithout_I)
	{
		/*Mat chords = Mat::zeros(m_canny.rows, m_canny.cols, CV_8UC3);
		Arc arcII = m_arcsInCoordinateQuarters[1][triplet[0]];
		Arc arcIII = m_arcsInCoordinateQuarters[2][triplet[1]];
		Arc arcIV = m_arcsInCoordinateQuarters[3][triplet[2]];
		drawArc(arcII, chords, arcColor[1]);
		drawArc(arcIII, chords, arcColor[2]);
		drawArc(arcIV, chords, arcColor[3]);
		
		// ищем прямые, проходящие через середины параллельных хорд
		// они пересекаются в центре эллипса
		auto line_34 = findLineCrossingMidpointBetweenHorizontalArcs(arcIII, arcIV);
		auto line_43 = findLineCrossingMidpointBetweenHorizontalArcs(arcIV, arcIII);
		Point intersection_1 = findIntersection(line_34, line_43);
		std::cout << intersection_1 << std::endl;
		auto line_23 = findLineCrossingMidpointBetweenUpperAndLowerArcs(arcII, arcIII);
		auto line_32 = findLineCrossingMidpointBetweenLowerAndUpperArcs(arcIII, arcII);
		Point intersection_2 = findIntersection(line_23, line_32);

		double q1, q2, q3, q4;
		q1 = static_cast<double>((arcIV[arcIV.size()-1].y - arcIII[arcIII.size()/2].y)) / 
			 (arcIV[arcIV.size()-1].x - arcIII[arcIII.size()/2].x);
		q2 = std::get<0>(line_43);
		q3 = static_cast<double>((arcIII[arcIII.size()-1].y - arcII[arcII.size()/2].y)) / 
			 (arcIII[arcIII.size()-1].x - arcII[arcII.size()/2].x);
		q4 = std::get<0>(line_32);

		double N, K, ro;
		std::tie(N, K, ro) = getEllipseParams(q1, q2, q3, q4);
		 
		Point center((intersection_1.x + intersection_2.x) / 2,
					 (intersection_1.y  + intersection_2.y) / 2);
		std::cout << "Center: " << center << std::endl;
			
		int A = abs(findMajorAxis(arcII, arcIII, arcIV, center, N, K));
		double B = A * N;
		double roInDegrees = ro/M_PI * 180;
		std::cout << "roInDegrees " << roInDegrees << std::endl;
		std::cout << "A " << A << " B " << B << std::endl;
		ellipse(chords, center, Size(A, B), roInDegrees, 0, 360, Scalar(0, 0, 255));
		displayImage("Chords", chords);*/
	}
}