#include "EllipseDetector.h"
#include <list>
#include <iostream>
#pragma once

using cv::Mat;
using cv::Point;

typedef std::vector<Point> Arc;
typedef std::vector<Arc> Arcs;
typedef std::vector<Ellipse> Ellipses;

// TODO: replace this class with tuples after switching to C++14
class Triplet
{
public:
	Triplet(int first, int second, int third)  
	{
		m_array = new int[3];
		m_array[0] = first;
		m_array[1] = second;
		m_array[2] = third;
	}
	int operator [] (int i) const
	{
		if(i >= 0 && i < 3)
			return m_array[i];
		else
			throw 0;
	}
private:
	int* m_array;
};

class FornaciariPratiDetector : public EllipseDetector
{
public:

	FornaciariPratiDetector(double scoreThreshold, double similarityWithLineThreshold, 
		double minimumAboveUnderAreaDifferenceRatio,
		int blurKernelSize = 5, int blurSigma = 1, int sobelKernelSize = 3);
	FornaciariPratiDetector(string configFile);
	// EllipseDetector functions
	virtual vector<Ellipse> DetectEllipses(const Mat& src);
	virtual vector<Ellipse> DetailedEllipseDetection(const Mat& src);
private:
	void getSobelDerivatives(const Mat& src);
	void useCannyDetector();
	void heuristicSearchOfArcs();
	void choosePossibleTriplets();
	inline bool isEdgePoint(Mat& edges, const Point& point);
	// dx shows whether this arc goes down to the right (I or III quarters) or left (II or IV quarters)
	void findArcThatIncludesPoint(int i_x, int i_y, int i_dx, Arc& o_arc);
	int calculateSquareUnderArc(const Arc& arc) const;
	double scoreForEllipse(const vector<Point>& ellipsePoints) const;
	double scoreForEllipse_2(const vector<Point>& ellipsePoints, const Arc& arc1,
							 const Arc& arc2, const Arc& arc3) const;
	bool curvatureCondition(const Arc& firstArc, const Arc& secondArc);
	void testTriplets();
	void blurEdges();
	bool getScore(const vector<Point>& ellipsePoints, const Arc& arc1,
					  const Arc& arc2, const Arc& arc3);
	bool isSimilar(const Ellipse& e1, const Ellipse& e2) const;
	void ellipsesClustering();
private:
	int m_sobelKernelSize;
	Mat m_sobelX, m_sobelY; // should use CS_16S
	Mat m_edges;
	int m_imgWidth;
	int m_imgHeight;
	Mat m_edgesCopy;
	Mat m_blurredEdges;
	int m_blurKernelSize;
	int m_blurSigma;
	Mat m_blurred; // TODO: delete it after implementing my own canny detector
	Arcs m_arcsInCoordinateQuarters[4];
	// содержат пары индексов совместимых арок в m_arcsInCoordinateQuarters[i]
	vector<std::pair<int, int>> m_possibleIandII, m_possibleIIandIII, m_possibleIIIandIV, m_possibleIVandI;
	// содержат тройки индексов совместимых арок
	vector<Triplet> m_tripletsWithout_I, m_tripletsWithout_II, m_tripletsWithout_III, m_tripletsWithout_IV;
	// середины арок
	vector<std::pair<int, int>> m_arcsMidPoints[4];
	Ellipses m_allDetectedEllipses;
	Ellipses m_uniqueEllipses;
	// для отладки
	std::set<int> m_remainingArcsIdx[4];

	// consts for thresholding
	const double SIMILARITY_WITH_LINE_THRESHOLD;
	const double MINIMUM_ABOVE_UNDER_AREA_DIFFERENCE_RATIO;
	const double SCORE_THRESHOLD;
};