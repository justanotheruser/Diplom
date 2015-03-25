#include "EllipseDetector.h"
#include <list>
#include <tuple>
#pragma once

using cv::Mat;
using cv::Point;

typedef std::vector<Point> Arc;
typedef std::vector<Arc> Arcs;

class FornaciariPratiDetector : public EllipseDetector
{
public:
	FornaciariPratiDetector(double similarityWithLineThreshold, double minimumAboveUnderAreaDifferenceRatio);
	FornaciariPratiDetector(string configFile);
	// EllipseDetector functions
	virtual vector<Ellipse> DetectEllipses(const Mat& src);
	virtual vector<Ellipse> DetailedEllipseDetection(const Mat& src);
private:
	void getSobelDerivatives(const Mat& src);
	void useCannyDetector();
	void heuristicSearchOfArcs();
	void choosePossibleTriplets();
	inline bool isEdgePoint(const Point& point);
	// dx shows whether this arc goes down to the right (I or III quarters) or left (II or IV quarters)
	void findArcThatIncludesPoint(int i_x, int i_y, int i_dx, Arc& o_arc);
	int calculateSquareUnderArc(const Arc& arc) const;
	void findMidPoints();
private:
	Mat m_sobelX, m_sobelY; // should use CS_16S
	Mat m_canny;
	Mat m_blurred; // TODO: delete it after implementing my own canny detector
	Arcs m_arcsInCoordinateQuarters[4];
	// содержат пары индексов совместимых арок в m_arcsInCoordinateQuarters[i]
	vector<std::pair<int, int>> m_possibleIandII, m_possibleIIandIII, m_possibleIIIandIV, m_possibleIVandI;
	// содержат тройки индексов совместимых арок
	vector<std::tuple<int, int, int>> m_tripletsWithout_I, m_tripletsWithout_II, m_tripletsWithout_III, m_tripletsWithout_IV;
	// середины арок
	vector<std::pair<int, int>> m_arcsMidPoints[4];

	// consts for thresholding
	const double SIMILARITY_WITH_LINE_THRESHOLD;
	const double MINIMUM_ABOVE_UNDER_AREA_DIFFERENCE_RATIO;
};