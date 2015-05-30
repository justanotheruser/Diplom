#include "HoughTransformAccumulator.h"


void HoughTransformAccumulator::Add(int value)
{
	if(m_points.size() < value+1)
	{
		m_points.resize(value+1);
	}
	++m_points[value];
}

int HoughTransformAccumulator::FindMax()
{
	int maxHit = 0;
	for (int i = 1; i < m_points.size(); i++)
	{
		if (m_points[maxHit] < m_points[i])
		{
			maxHit = i;
		}
	}
	return maxHit;
}