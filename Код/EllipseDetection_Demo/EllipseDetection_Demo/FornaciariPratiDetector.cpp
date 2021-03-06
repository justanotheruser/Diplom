#include "FornaciariPratiDetector.h"
#include <list>
#include "opencv2\highgui\highgui.hpp"

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

vector<Ellipse> FornaciariPratiDetector::DetectEllipses(const Mat& src)
{
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
	// TODO: ����������� �������� ����� ������ � ������������������ ����������� �������
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
	for (int i = 0; i < 4; i++)
		arcsInCoordinateQuarters[i].resize(50);
	//Arcs allArcs(100);
	const int lineSimilarityThreshold = 6;

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
				if (newArc.size() <= 16) // �������� ������� ��������� ����
					continue;
				int rightX = newArc.front().x;
				int rightY = newArc.front().y;
				int leftX = newArc.back().x;
				int leftY = newArc.back().y;
				// �������� ������� ��������� (����������� �� ������������ ��� �������������� ������)
				double width = rightX - leftX;
				double height = abs(rightY - leftY);
				if(width/height >= lineSimilarityThreshold || height/width >= lineSimilarityThreshold)
					continue;
				// ���� ������� ��� ������ ���������� �� ������� ��� ������ ������,
				// ��� �� eps * totalSquare, �� ����������� ��� ������ ��� �� �������� � �� ��������
				// (������� ������� �� ������)
				double eps = 0.1;
				if(rightY > leftY){ // I ��� III ��������
					// ������� ������� ��� � ��� ������
					int underSquare = 0, totalSquare = (rightX - leftX) * abs(rightY - leftY);
					auto i = newArc.begin();
					int prev_x = i->x;
					i++;
					for(;i != newArc.end(); i++){
						if(i->x != prev_x){ // ���� � ������ �������� ������ y, �� �������� ������� ���� ������� ������
							underSquare += rightY - i->y;
							prev_x = i->x;
						}
					}
					if(underSquare - (totalSquare - underSquare) > eps*totalSquare) // �������� ����� ������ - I 
						arcsInCoordinateQuarters[0].push_back(newArc);
					else if(underSquare - (totalSquare - underSquare) < -eps*totalSquare)  // �������� ���� ������ - III
						arcsInCoordinateQuarters[2].push_back(newArc);

				}
				else{ // II ��� IV 
					int underSquare = 0, totalSquare = abs((rightX - leftX) * (leftY - rightY));
					auto i = newArc.begin();
					int prev_x = i->x;
					i++;
					for(;i != newArc.end(); i++){
						if(i->x != prev_x){ // ���� � ������ �������� ������ y, �� �������� ������� ���� ������� ������
							underSquare += leftY - i->y;
							prev_x = i->x;
						}
					}
					if(underSquare - (totalSquare - underSquare) > eps*totalSquare) // �������� ����� ������ - II
						arcsInCoordinateQuarters[1].push_back(newArc);
					else if(underSquare - (totalSquare - underSquare) < -eps*totalSquare) // �������� ���� ������ - IV
						arcsInCoordinateQuarters[3].push_back(newArc);
				}
				//allArcs.push_back(newArc);
			}
		}
	}

	Mat result = Mat::zeros(m_canny.rows, m_canny.cols, CV_8UC3);
	uchar arcColor[4][3] = {{0, 0, 255}, {0, 255, 0}, {255, 0, 0}, {0, 255, 255}};
	for(int i = 0; i < 4; i++)
		for (auto arc : arcsInCoordinateQuarters[i])
			drawArc(arc, result, arcColor[i]);
	displayImage("All arcs", result); 
}

Arc FornaciariPratiDetector::findArcThatIncludesPoint(int col, int row, short* sX, short* sY)
{
	Arc arc;
	arc.emplace_back(col, row);
	m_canny.at<uchar>(Point(col, row)) = 0;
	// ���������� ����������� ��������� (�������� II � IV �� I � III)
	if(*sX > 0 && *sY > 0 || *sX < 0 && *sY < 0)
	{
		Point newPoint, lastMostLeft;
		do // ���� ����� ����� ���� �����-����� ���� �� ���������� ��������
		{
			newPoint = lastMostLeft = arc.back();
			// ���� ������� � ������ �������, �� � ���� ����� ��� ������ � ������������ �������
			if (lastMostLeft.y + 1 <= m_canny.rows-1)
			{
				// ������ ����
				newPoint = lastMostLeft + Point(0, 1);
				if (isEdgePoint(newPoint))
					arc.push_back(newPoint);
				else if (lastMostLeft.x - 1 >= 0)
				{
					// ������������ �����
					newPoint = lastMostLeft + Point(-1, 1);
					if (isEdgePoint(newPoint))
						arc.push_back(newPoint);
					else
					{
						// ������ �����
						newPoint = lastMostLeft + Point(-1, 0);
						if (isEdgePoint(newPoint))
							arc.push_back(newPoint);
					}
				}
			}
			else if (lastMostLeft.x - 1 >= 0)
			{
				// ������ �����
				newPoint = lastMostLeft + Point(-1, 0);
				if (isEdgePoint(newPoint))
					arc.push_back(newPoint);
			}
		} while(lastMostLeft != arc.back());
	}
	else
	{
		Point newPoint, lastMostRight;
		do // ���� ����� ����� ���� ������-����� ���� �� ���������� ��������
		{
			newPoint = lastMostRight = arc.front();
			// ���� ������� � ������ �������, �� � ���� ����� ��� ������ � ������������ �������
			if (lastMostRight.y + 1 <= m_canny.rows-1)
			{
				// ������ ����
				newPoint = lastMostRight + Point(0, 1);
				if (isEdgePoint(newPoint))
					arc.push_front(newPoint);
				else if (lastMostRight.x + 1 <= m_canny.cols-1)
				{
					// ������������ �����
					newPoint = lastMostRight + Point(1, 1);
					if (isEdgePoint(newPoint))
						arc.push_front(newPoint);
					else
					{
						// ������ �����
						newPoint = lastMostRight + Point(1, 0);
						if (isEdgePoint(newPoint))
							arc.push_front(newPoint);
					}
				}
			}
			else if (lastMostRight.x + 1 <= m_canny.cols-1)
			{
				// ������ �����
				newPoint = lastMostRight + Point(1, 0);
				if (isEdgePoint(newPoint))
					arc.push_front(newPoint);
			}
		} while(lastMostRight != arc.front());
	}
	return arc;
}

Mat FornaciariPratiDetector::findArcs(const Mat& src){
	vector<list<Point>> arcs[4];
	Mat result = Mat::zeros(m_canny.rows, m_canny.cols, CV_8UC3);

	// ������������ � ���� ������ ��� ��������� ������, ����������� �� �����
	/*const int lineSimilarityThreshold = 20; // ���������� ����� ����� ������, � ������� ���� �� ��������� ���������

	for(int row = 0; row < canny.rows; row++){ 
		uchar* p = canny.ptr(row);
		short* sX = sobelX.ptr<short>(row);
		short* sY = sobelY.ptr<short>(row);
		for(int col = 0; col < canny.cols; col++, p++, sX++, sY++) {
			if(*p == 255 && *sX != 0){
				// ���������� ����������� ��������� (��������  II � IV �� I � III)
				int dx, dy; // ��������, ������������ � ��������� ������
				int yBorder, sYsXSign;
				if(*sY * *sX > 0){ // ���� ����� ��, ��� � *sY / *sX, �� ��������� ���� �������
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

				// ���� ��� ������������� ���� ���� �����
				int rightX, rightY, leftX, leftY;
				int newRightX = col, newRightY = row;
				// ���������� ��� �������� �� ���������
				int looksLikeVerticalLine = 0, looksLikeHorizontalLine = 0;
				do{  // ���� ����� ������ ������ �� ��������
					rightX = newRightX; rightY = newRightY;
					uchar* h = canny.ptr(rightY + dy); // ����� �� ���������
					h += rightX + dx;
					if(*h == 255){
						short* new_sX = sobelX.ptr<short>(rightY + dy); // ���������, ��� ����������� ��������� � ���� ����� ������� ��� ��
						new_sX += rightX + dx;
						short* new_sY = sobelY.ptr<short>(rightY + dy);
						new_sY += rightX + dx;
						if(sYsXSign * (*new_sY) * (*new_sX) > 0){ // ����� ���������, �� ������
							newRightY = rightY + dy;
							newRightX = rightX + dx;
							looksLikeVerticalLine = 0;	looksLikeHorizontalLine = 0;
							new_arc.push_front(Point(newRightX, newRightY));
							canny.at<uchar>(Point(newRightX, newRightY)) = 0;
							continue;
						}
					}
					h = canny.ptr(rightY + dy); // ��������� ������ y
					h += rightX;
					if(*h == 255){
						// ���������, ��� ����� ������� x
						looksLikeVerticalLine++;
						// ���� ������� �����, �� ���� ������ ���������� ������������ � ������������ ������ � ����� �������� �
						// ���� ���, �� ����������
						if(looksLikeVerticalLine < lineSimilarityThreshold){
							short* new_sX = sobelX.ptr<short>(rightY + dy); // ���������, ��� ����������� ��������� � ���� ����� ������� ��� ��
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
					h = canny.ptr(rightY); // ��������� ������ �
					h += rightX + dx;
					if(*h == 255){
						// ���������, ��� ����� ������� y
						looksLikeHorizontalLine++;
						if(looksLikeHorizontalLine < lineSimilarityThreshold){
							short* new_sX = sobelX.ptr<short>(rightY); // ���������, ��� ����������� ��������� � ���� ����� ������� ��� ��
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
				// ��������� ���� �������/������ ������� �� ���������� �������� ��� �� ������ � ���� �����������
				}while(!(newRightX==rightX && newRightY==rightY) && newRightX != canny.cols-1 && newRightY != yBorder);

				yBorder = yBorder==0 ? canny.rows - 1 : 0;
				int newLeftX = col, newLeftY = row;
				looksLikeVerticalLine = 0; looksLikeHorizontalLine = 0;
				do{
					leftX = newLeftX; leftY = newLeftY;
					uchar* l = canny.ptr(leftY - dy); // ����� �� ���������
					l += leftX - dx;
					if(*l == 255){
						short* new_sX = sobelX.ptr<short>(leftY - dy); // ���������, ��� ����������� ��������� � ���� ����� ������� ��� ��
						new_sX += leftX - dx;
						short* new_sY = sobelY.ptr<short>(leftY - dy);
						new_sY += leftX - dx;
						if(sYsXSign * (*new_sY) * (*new_sX) > 0){ // ����� ���������, �� ������
							newLeftX = leftX - dx;
							newLeftY = leftY - dy;
							looksLikeVerticalLine = 0;	looksLikeHorizontalLine = 0;
							new_arc.push_back(Point(newLeftX, newLeftY));
							canny.at<uchar>(Point(newLeftX, newLeftY)) = 0;
							continue;
						}
					}
					l = canny.ptr(leftY - dy); // ��������� ������ y
					l += leftX;
					if(*l == 255){
						// ���������, ��� ����� ������� x
						looksLikeVerticalLine++;
						if(looksLikeVerticalLine < lineSimilarityThreshold){
							short* new_sX = sobelX.ptr<short>(leftY - dy); // ���������, ��� ����������� ��������� � ���� ����� ������� ��� ��
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


					l = canny.ptr(leftY); // ��������� ������ �
					l += leftX - dx;
					if(*l == 255){
						// ���������, ��� ����� ������� y
						looksLikeHorizontalLine++;
						if(looksLikeHorizontalLine < lineSimilarityThreshold){
							short* new_sX = sobelX.ptr<short>(leftY); // ���������, ��� ����������� ��������� � ���� ����� ������� ��� ��
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
				// ��������� ���� ������/����� ������� �� ���������� �������� ��� �� ������ � ���� �����������
				}while(!(newLeftX==leftX && newLeftY==leftY) && newLeftX != 0 && newLeftY != yBorder);
					
				// �������� ������� �������� ����
				if(new_arc.size() <= 16)
					continue;
				// �������� ������� ��������� (����������� �� ������������ ��� �������������� ������)
				double width = rightX - leftX;
				double height = abs(rightY - leftY);
				if(width/height >= lineSimilarityThreshold || height/width >= lineSimilarityThreshold)
					continue;

				// ���� ������� ��� ������ ���������� �� ������� ��� ������ ������,
				// ��� �� eps * totalSquare, �� ����������� ��� ������ ��� �� �������� � �� ��������
				// (������� ������� �� ������)
				double eps = 0.1;
				if(rightY > leftY){ // I ��� III ��������
					// ������� ������� ��� � ��� ������
					int underSquare = 0, totalSquare = abs((rightX - leftX) * (rightY - leftY));
					auto i = new_arc.begin();
					int prev_x = i->x;
					i++;
					for(;i != new_arc.end(); i++){
						if(i->x != prev_x){ // ���� � ������ �������� ������ y, �� �������� ������� ���� ������� ������
							underSquare += rightY - i->y;
							prev_x = i->x;
						}
					}
					if(underSquare - (totalSquare - underSquare) > eps*totalSquare) // �������� ����� ������ - I 
						arcs[0].push_back(new_arc);
					else if(underSquare - (totalSquare - underSquare) < -eps*totalSquare)  // �������� ���� ������ - III
						arcs[2].push_back(new_arc);

				}
				else{ // II ��� IV 
					int underSquare = 0, totalSquare = abs((rightX - leftX) * (leftY - rightY));
					auto i = new_arc.begin();
					int prev_x = i->x;
					i++;
					for(;i != new_arc.end(); i++){
						if(i->x != prev_x){ // ���� � ������ �������� ������ y, �� �������� ������� ���� ������� ������
							underSquare += leftY - i->y;
							prev_x = i->x;
						}
					}
					if(underSquare - (totalSquare - underSquare) > eps*totalSquare) // �������� ����� ������ - II
						arcs[1].push_back(new_arc);
					else if(underSquare - (totalSquare - underSquare) < -eps*totalSquare) // �������� ���� ������ - IV
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