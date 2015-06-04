#include "HoughTransformAccumulator.h"
#include <iostream>

void HoughTransformAccumulator::Add(double value)
{
	auto it = m_points.find(value);
	if(it != m_points.end())
	{
		it->second++;
	}
	else
	{
		m_points.emplace(value, 0);
	}
}

double HoughTransformAccumulator::FindMax()
{
	auto max = m_points.begin();
	for (auto it = ++m_points.begin(); it != m_points.end(); ++it)
	{
		if (it->second > max->second)
		{
			max = it;
		}
	}
	return max->first;
}