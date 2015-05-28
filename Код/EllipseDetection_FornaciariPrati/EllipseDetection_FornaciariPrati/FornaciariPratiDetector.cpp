#include "FornaciariPratiDetector.h"
#include "opencv2\highgui\highgui.hpp"
#include <tuple>

using namespace cv;
using std::string;
using std::list;

uchar arcColor[4][3] = {{0, 0, 255}, {0, 255, 0}, {255, 0, 0}, {0, 255, 255}};

double getSlope(const std::vector<Point>& midPoints)
{
	int middle = midPoints.size() / 2;
	double slope = 0;
	for (unsigned int i = 0; i + middle < midPoints.size(); i++)
	{
	   double xDiff = midPoints[i].x - midPoints[i + middle].x;
	   double yDiff = midPoints[i].y - midPoints[i + middle].y;
	   slope +=  yDiff / xDiff;
	}
	slope /= std::ceil(midPoints.size() / 2.);
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

// ������� ��������� ������, ���������� ����� �������� ����, ������������ �����, ��������� �����
// ��������� ������ ���� � ������ ������ ������
std::tuple<double, int> findLineCrossingMidpointBetweenMidAndLowPoints(const Arc& arcWithMid, const Arc& arcWithLow)
{
	std::vector<Point> midPoints;
	// a) ��������� ������ ����� ������ ����� ������ ���� � ��������� ������
	Point Pa = arcWithMid[arcWithMid.size()/2];
	Point Hb = arcWithLow[arcWithLow.size()-1];
	Point dir_vector = Pa - Hb; // ������������ ������ �������� �����
	midPoints.emplace_back((Pa.x + Hb.x) / 2, (Pa.y + Hb.y) / 2);
	// �) ��������� ��� 5 ������������ ��
	// ��������� ������ ����� ������� ������ ������� ���� ������ ����
	// ������� ����� ���������� ������� �� 6 ������
	int delta = arcWithMid.size()/4;
	// ������ ����������� ����� ����� ���������� ������ ���� ���� � ����, ������� ����������, ������ ��� 
	// ������� ������ ��������� �����
	int checkpoint = arcWithLow.size()-1;
	
	for (int point = arcWithMid.size()/2 - delta; point >= 0; point -= delta)
	{
		// ���� ����� �� ������ ����, ������� ���������� �����
		// ����� ����� ��, ������� ����� ����������� � ��������� ������ � ������ ������������� � �
		// ������ ������������ �������� ��� ���������� ������� (���������� �� 0).
		// ����������� � ��������� ������, ���������� ����� ������� ����� ������ ����
		int � = dir_vector.x * arcWithMid[point].y - dir_vector.y * arcWithMid[point].x; 
		int prev_discrepancy = abs(dir_vector.y * arcWithLow[checkpoint].x - dir_vector.x * arcWithLow[checkpoint].y + �);
		for (checkpoint--; checkpoint >=0; checkpoint--)
		{
			int discrepancy = abs(dir_vector.y * arcWithLow[checkpoint].x - dir_vector.x * arcWithLow[checkpoint].y + �);
			if (discrepancy > prev_discrepancy)
			{
				checkpoint++;
				break;
			}
			prev_discrepancy = discrepancy;
		}
		// �) ����� �������� ����
		midPoints.emplace_back((arcWithMid[point].x + arcWithLow[checkpoint].x) / 2, 
							   (arcWithMid[point].y + arcWithLow[checkpoint].y) / 2);
	}
	double slope = getSlope(midPoints);
	Point pointOnLineCrossingCenter = getAveragePoint(midPoints);
	// y = slope*x + coeff
	int coeff = pointOnLineCrossingCenter.y - slope * pointOnLineCrossingCenter.x;
	return std::make_tuple(slope, coeff);
}

std::tuple<double, int> findLineCrossingMidpointBetweenMidAndHighPoints(const Arc& arcWithMid, const Arc& arcWithHigh)
{
	std::vector<Point> midPoints;
	// a) ��������� ������ ����� ������ ����� ������ ���� � ��������� ������
	Point Pa = arcWithMid[arcWithMid.size()/2];
	Point Hb = arcWithHigh[0];
	Point dir_vector = Pa - Hb; // ������������ ������ �������� �����
	midPoints.emplace_back((Pa.x + Hb.x) / 2, (Pa.y + Hb.y) / 2);
	// �) ��������� ��� 5 ������������ ��
	// ��������� ������ ����� ������� ������ ������� ���� ������ ����
	// ������� ����� ���������� ������� �� 6 ������
	int delta = arcWithMid.size()/48;
	// ������ ����������� ����� ����� ���������� ������ ���� ���� � ����, ������� ����������, ������ ��� 
	// ������� ������ ��������� �����
	int checkpoint = 0;
	
	for (int point = arcWithMid.size()/2 - delta; point >= 0; point -= delta)
	{
		// ���� ����� �� ������ ����, ������� ���������� �����
		// ����� ����� ��, ������� ����� ����������� � ��������� ������ � ������ ������������� � �
		// ������ ������������ �������� ��� ���������� ������� (���������� �� 0).
		// ����������� � ��������� ������, ���������� ����� ������� ����� ������ ����
		int � = dir_vector.x * arcWithMid[point].y - dir_vector.y * arcWithMid[point].x; 
		int prev_discrepancy = abs(dir_vector.y * arcWithHigh[checkpoint].x - dir_vector.x * arcWithHigh[checkpoint].y + �);
		for (checkpoint++; checkpoint < arcWithHigh.size(); checkpoint++)
		{
			int discrepancy = abs(dir_vector.y * arcWithHigh[checkpoint].x - dir_vector.x * arcWithHigh[checkpoint].y + �);
			if (discrepancy > prev_discrepancy)
			{
				checkpoint--;
				break;
			}
			prev_discrepancy = discrepancy;
		}
		// �) ����� �������� ����
		midPoints.emplace_back((arcWithMid[point].x + arcWithHigh[checkpoint].x) / 2, 
							   (arcWithMid[point].y + arcWithHigh[checkpoint].y) / 2);
	}
	double slope = getSlope(midPoints);
	Point pointOnLineCrossingCenter = getAveragePoint(midPoints);
	// y = slope*x + coeff
	int coeff = pointOnLineCrossingCenter.y - slope * pointOnLineCrossingCenter.x;
	return std::make_tuple(slope, coeff);
}

void findCenterIandII(const Arc& arcI, const Arc& arcII)
{

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
	useCannyDetector(); // TODO: ����������� �������� ����� ������ � ������������������ ����������� �������
	heuristicSearchOfArcs();
	choosePossibleTriplets();
	testTriplets();
	waitKey(0);
	vector<Ellipse> ellipses;
	return ellipses;
}

vector<Ellipse> FornaciariPratiDetector::DetailedEllipseDetection(const Mat& src)
{
	// �������� �����������
	//displayImage("Source", src);

	// TODO: ������� ��������� ������ ����������
	// http://www.youtube.com/watch?v=lC-IrZsdTrw
	Mat canny, sobelX, sobelY;

	getSobelDerivatives(src);
	
	useCannyDetector();

	// TODO: �������� �������������� ������ ������ ����:
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
	// �������� ������ �������, ����� �������� ����������� � ��������
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
				Point upperRight, bottomLeft; // ������� �����, ������ �������� ���� ����
				upperRight.y = row;
				// ���������� ����������� ���������
				if(*sX > 0 && *sY > 0 || *sX < 0 && *sY < 0) // II ��� IV
				{
					findArcThatIncludesPoint(col, row, -1, newArc);
					upperRight.x = col;
					bottomLeft = newArc.back();
					if (newArc.size() <= 16) // �������� ������� ��������� ����
						continue;
					// �������� ������� ��������� (����������� �� ������������ ��� �������������� ������)
					double width  = upperRight.x - bottomLeft.x;
					double height = bottomLeft.y - upperRight.y;
					if(width/height >= SIMILARITY_WITH_LINE_THRESHOLD || height/width >= SIMILARITY_WITH_LINE_THRESHOLD)
						continue;
					// ���� ������� ��� ������ ���������� �� ������� ��� ������ ������,
					// ��� �� totalSquare * MINIMUM_ABOVE_UNDER_AREA_DIFFERENCE_RATIO, �� ����������� ��� ������ ��� 
					// �� �������� � �� �������� (������� ������� �� ������)
					int totalSquare = width * height, underSquare = calculateSquareUnderArc(newArc);
					int underMinusAbove = underSquare - (totalSquare - underSquare);
					// �������� ����� ������ - II
					if(underMinusAbove > MINIMUM_ABOVE_UNDER_AREA_DIFFERENCE_RATIO * totalSquare) 
						m_arcsInCoordinateQuarters[1].push_back(newArc);
					// �������� ���� ������ - IV
					else if(underMinusAbove < - MINIMUM_ABOVE_UNDER_AREA_DIFFERENCE_RATIO * totalSquare) 
						m_arcsInCoordinateQuarters[3].push_back(newArc);
				}
				else // I ��� III
				{
					findArcThatIncludesPoint(col, row, 1, newArc);
					upperRight.x = newArc.back().x;
					bottomLeft.x = col; bottomLeft.y = newArc.back().y;
					if (newArc.size() <= 16) // �������� ������� ��������� ����
						continue;
					// �������� ������� ��������� (����������� �� ������������ ��� �������������� ������)
					double width  = upperRight.x - bottomLeft.x;
					double height = bottomLeft.y - upperRight.y;
					if(width/height >= SIMILARITY_WITH_LINE_THRESHOLD || height/width >= SIMILARITY_WITH_LINE_THRESHOLD)
						continue;
					// ���� ������� ��� ������ ���������� �� ������� ��� ������ ������,
					// ��� �� totalSquare * MINIMUM_ABOVE_UNDER_AREA_DIFFERENCE_RATIO, �� ����������� ��� ������ ��� 
					// �� �������� � �� �������� (������� ������� �� ������)
					int totalSquare = width * height, underSquare = calculateSquareUnderArc(newArc);
					int underMinusAbove = underSquare - (totalSquare - underSquare);
					// �������� ����� ������ - I 
					if(underMinusAbove > MINIMUM_ABOVE_UNDER_AREA_DIFFERENCE_RATIO * totalSquare) 
						m_arcsInCoordinateQuarters[0].push_back(newArc);
					// �������� ���� ������ - III
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
	do // ���� ����� ����� ���� c����-�� ���������-����� ���� �� ���������� ��������
	{
		newPoint = lastPoint = arc.back();
		// ���� ������� � ������ �������, �� � ���� ����� ��� ������ � ������������ �������
		if (lastPoint.y + 1 <= m_canny.rows-1)
		{
			// ������ ����
			newPoint = lastPoint + Point(0, 1);
			if (isEdgePoint(newPoint))
				arc.push_back(newPoint);
			else if (lastPoint.x - 1 >= 0)
			{
				// ������������ �����
				newPoint = lastPoint + Point(dx, 1);
				if (isEdgePoint(newPoint))
					arc.push_back(newPoint);
				else
				{
					// ������ �� ����������� 
					newPoint = lastPoint + Point(dx, 0);
					if (isEdgePoint(newPoint))
						arc.push_back(newPoint);
				}
			}
		}
		else if (lastPoint.x - 1 >= 0)
		{
			// ������ �� �����������
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
		if(i->x != prev_x) // ���� � ������ �������� ������ y, �� �������� ������� ���� ������� ������
		{
			underSquare += bottomY - i->y;
			prev_x = i->x;
		}
	}
	return underSquare;
}

void FornaciariPratiDetector::choosePossibleTriplets(){
	// ����������, �.�. �������� �������� ������� � ����� �� �������� �������
	int numArcsI = m_arcsInCoordinateQuarters[0].size();
	int numArcsII = m_arcsInCoordinateQuarters[1].size();
	int numArcsIII = m_arcsInCoordinateQuarters[2].size();
	int numArcsIV = m_arcsInCoordinateQuarters[3].size();
	// �������� ������ �������, ����� �������� ����������� � ��������
	for (int i = 0; i < 4; i++)
	{
		m_possibleIandII.clear();   m_possibleIandII.reserve(numArcsI*numArcsII);
		m_possibleIIandIII.clear(); m_possibleIIandIII.reserve(numArcsII*numArcsIII);
		m_possibleIIIandIV.clear();	m_possibleIIIandIV.reserve(numArcsIII*numArcsIV);
		m_possibleIVandI.clear();   m_possibleIVandI.reserve(numArcsIV*numArcsI);
	}
	// ������� �������� ���������� ����
	Mat curvatureConditionMat = Mat::zeros(m_canny.rows, m_canny.cols, CV_8UC3);
	int cond1 = 0, cond2 = 0;
	for(int aI = 0; aI < numArcsI; aI++){ 
		for(int aII = 0; aII < numArcsII; aII++)
		{
			// aI ������ ���� ������ aII
			if(m_arcsInCoordinateQuarters[1][aII].front().x <= m_arcsInCoordinateQuarters[0][aI].front().x)
			{
				if(curvatureCondition(m_arcsInCoordinateQuarters[0][aI], m_arcsInCoordinateQuarters[1][aII]))
				{
					m_possibleIandII.emplace_back(aI, aII);
				}
				/*else
				{
					cond1++;
					if (cond1 <= 3)
					{
						uchar randomColor[3] = {rand() % 256, rand() % 256, rand() % 256};
						drawArc(m_arcsInCoordinateQuarters[0][aI], curvatureConditionMat, randomColor);
						drawArc(m_arcsInCoordinateQuarters[1][aII], curvatureConditionMat, randomColor);
					}
				}*/
			}
		}
	}
	for(int aII = 0; aII < numArcsII; aII++){
		for(int aIII = 0; aIII < numArcsIII; aIII++)
		{
			// aIII ������ ���� ��� aII
			if(m_arcsInCoordinateQuarters[2][aIII].front().y >= m_arcsInCoordinateQuarters[1][aII].back().y)
			{
				
				if(curvatureCondition(m_arcsInCoordinateQuarters[1][aII], m_arcsInCoordinateQuarters[2][aIII]))
					m_possibleIIandIII.emplace_back(aII, aIII);
				/*else 
				{
					
					if (cond1 == 3)
					{
						/*uchar randomColor[3] = {rand() % 256, rand() % 256, rand() % 256};
						drawArc(m_arcsInCoordinateQuarters[1][aII], curvatureConditionMat, randomColor);
						randomColor[0] = randomColor[1] = randomColor[2] = rand() % 256;
						drawArc(m_arcsInCoordinateQuarters[2][aIII], curvatureConditionMat, randomColor);
					}
					
				}*/
			}
		}
	}
	for(int aIII = 0; aIII < numArcsIII; aIII++){ 
		for(int aIV = 0; aIV < numArcsIV; aIV++)
		{
			// aIII ������ ���� ����� aIV
			if(m_arcsInCoordinateQuarters[2][aIII].back().x <= m_arcsInCoordinateQuarters[3][aIV].back().x)
			{
				if(curvatureCondition(m_arcsInCoordinateQuarters[2][aIII], m_arcsInCoordinateQuarters[3][aIV]))
					m_possibleIIIandIV.emplace_back(aIII, aIV);
				/*else if (cond1 == 0)
				{
					/*uchar randomColor[3] = {rand() % 256, rand() % 256, rand() % 256};
					drawArc(m_arcsInCoordinateQuarters[3][aIV], curvatureConditionMat, randomColor);
					drawArc(m_arcsInCoordinateQuarters[2][aIII], curvatureConditionMat, randomColor);
					cond1++
				}*/
			}
		}
	}
	for(int aIV = 0; aIV < numArcsIV; aIV++){
		for(int aI = 0; aI < m_arcsInCoordinateQuarters[0].size(); aI++)
		{
			// aIV ������ ���� ���� aI
			if(m_arcsInCoordinateQuarters[3][aIV].front().y >= m_arcsInCoordinateQuarters[0][aI].back().y)
				if(curvatureCondition(m_arcsInCoordinateQuarters[3][aIV], m_arcsInCoordinateQuarters[0][aI]))
					m_possibleIVandI.emplace_back(aIV, aI);
				/*else if (cond1 == 0)
				{				
					/*uchar randomColor[3] = {rand() % 256, rand() % 256, rand() % 256};
					drawArc(m_arcsInCoordinateQuarters[0][aI], curvatureConditionMat, randomColor);
					drawArc(m_arcsInCoordinateQuarters[3][aIV], curvatureConditionMat, randomColor);
					cond1++;
				}*/
		}
	}


	// ������ ���������� ��������� ������
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
			// ���� ������� �� ������ �������� � ���� ��� ���� � ��� ��, �� ��� ��������� ������
			if(m_possibleIandII[i].second == m_possibleIIandIII[j].first)
				m_tripletsWithout_IV.emplace_back(m_possibleIandII[i].first, m_possibleIandII[i].second, m_possibleIIandIII[j].second);
		}
		for(int j = 0; j < numOfPairsIVandI; j++)
		{
			// ���� ������� �� ������ �������� � ���� ��� ���� � ��� ��, �� ��� ��������� ������
			if(m_possibleIandII[i].first == m_possibleIVandI[j].second)
				m_tripletsWithout_III.emplace_back(m_possibleIandII[i].first, m_possibleIandII[i].second, m_possibleIVandI[j].first);
		}
	}
	for(int i = 0; i < numOfPairsIIIandIV; i++)
	{
		for(int j = 0; j < numOfPairsIVandI; j++)
		{
			// ���� ������� �� �������� �������� � ���� ��� ���� � ��� ��, �� ��� ��������� ������
			if(m_possibleIIIandIV[i].second == m_possibleIVandI[j].first)
				m_tripletsWithout_II.emplace_back(m_possibleIVandI[j].second, m_possibleIIIandIV[i].first, m_possibleIIIandIV[i].second);
		}
		for(int j = 0; j < numOfPairsIIandIII; j++)
		{
			// ���� ������� �� ������� �������� � ���� ��� ���� � ��� ��, �� ��� ��������� ������
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
	const double thresholdCenterDiff = 5;
	Mat canvas = Mat::zeros(m_canny.rows, m_canny.cols, CV_8UC3);
	for(auto triplet : m_tripletsWithout_IV)
	{
		Mat chords = Mat::zeros(m_canny.rows, m_canny.cols, CV_8UC3);
		Arc arcI = m_arcsInCoordinateQuarters[0][triplet[0]];
		Arc arcII = m_arcsInCoordinateQuarters[1][triplet[1]];
		Arc arcIII = m_arcsInCoordinateQuarters[2][triplet[2]];
		drawArc(arcI, chords, arcColor[0]);
		drawArc(arcII, chords, arcColor[1]);
		drawArc(arcIII, chords, arcColor[2]);
		

		// 1) ��������� ������, ���������� ����� ������� ���� ����� ����� I � II
		int coeff_1, coeff_2;
		double slope_1, slope_2;
		std::tie(slope_1, coeff_1) = findLineCrossingMidpointBetweenMidAndLowPoints(arcI, arcII);
		std::tie(slope_2, coeff_2) = findLineCrossingMidpointBetweenMidAndLowPoints(arcII, arcI);
		line(chords, Point(0, coeff_1), Point(600, slope_1*600 + coeff_1), Scalar(255, 0, 0));
		line(chords, Point(0, coeff_2), Point(600, slope_2*600 + coeff_2), Scalar(255, 0, 0));

		std::tie(slope_1, coeff_1) = findLineCrossingMidpointBetweenMidAndLowPoints(arcI, arcIII);
		std::tie(slope_2, coeff_2) = findLineCrossingMidpointBetweenMidAndLowPoints(arcIII, arcI);
		line(chords, Point(0, coeff_1), Point(600, slope_1*600 + coeff_1), Scalar(0, 255, 0));
		line(chords, Point(0, coeff_2), Point(600, slope_2*600 + coeff_2), Scalar(0, 255, 0));

		std::tie(slope_1, coeff_1) = findLineCrossingMidpointBetweenMidAndLowPoints(arcII, arcIII);
		std::tie(slope_2, coeff_2) = findLineCrossingMidpointBetweenMidAndHighPoints(arcIII, arcI);
		line(chords, Point(0, coeff_1), Point(600, slope_1*600 + coeff_1), Scalar(0, 0, 255));
		line(chords, Point(0, coeff_2), Point(600, slope_2*600 + coeff_2), Scalar(0, 0, 255));

		displayImage("Chords", chords);
	}

	for(auto triplet : m_tripletsWithout_III)
	{
	}

	for(auto triplet : m_tripletsWithout_II)
	{
	}

	for(auto triplet : m_tripletsWithout_I)
	{
	}
}