#include "FornaciariPratiDetector.h"
#include <list>
#include "opencv2\highgui\highgui.hpp"

using namespace cv;
using std::string;
using std::list;

vector<Ellipse> FornaciariPratiDetector::DetectEllipses(const Mat& src)
{
	vector<Ellipse> ellipses;
	return ellipses;
}

vector<Ellipse> FornaciariPratiDetector::DetailedEllipseDetection(const Mat& src)
{
	namedWindow("Source image", CV_WINDOW_AUTOSIZE);
	imshow("Source image", src);
	waitKey(0);
	vector<Ellipse> ellipses;
	return ellipses;
}

Mat FornaciariPratiDetector::findArcs(const Mat& src){
	vector<list<Point>> arcs[4];

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
	// ������������ � ���� ������ ��� ��������� ������, ����������� �� �����
	const int lineSimilarityThreshold = 20; // ���������� ����� ����� ������, � ������� ���� �� ��������� ���������

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
		//drawArc(result, *aI, aI_color);
	}
	for(auto aII = arcs[1].begin(); aII != arcs[1].end(); aII++){
		//drawArc(result, *aII, aII_color);
	}
	for(auto aIII = arcs[2].begin(); aIII != arcs[2].end(); aIII++){
		//drawArc(result, *aIII, aIII_color);
	}
	for(auto aIV = arcs[3].begin(); aIV != arcs[3].end(); aIV++){
		//drawArc(result, *aIV, aIV_color);
	}
	return result;
}