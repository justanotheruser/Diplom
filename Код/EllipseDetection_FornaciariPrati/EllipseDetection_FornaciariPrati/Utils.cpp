#include "Utils.h"

// because 5*sqrt(2) ~= 7, so for each 5 diagonal steps from one end we should make
// 2 additional steps from another end
 Point findArcMiddlePoint(const std::vector<Point>& arc)
{
	auto lowerBound = arc.begin();
	auto upperBound = arc.end()-1;
	std::vector<Point>::const_iterator nextLowerBound, nextUpperBound;
	int lowerDiagonalCounter = 0;
	int upperDiagonalCounter = 0;
	while (upperBound > lowerBound)
	{
		nextLowerBound = lowerBound+1;
		if (nextLowerBound->x != lowerBound->x && nextLowerBound->y != lowerBound->y)
		{
			lowerDiagonalCounter++;
		}
		lowerBound = nextLowerBound;

		nextUpperBound = upperBound-1;
		if (nextUpperBound->x != upperBound->x && nextUpperBound->y != upperBound->y)
		{
			upperDiagonalCounter++;
		}
		upperBound = nextUpperBound;
		
		if (lowerDiagonalCounter >= 5)
		{
			for (int i = 0; i < 2 && upperBound > lowerBound; i++)
			{ 
				nextUpperBound = upperBound-1;
				if (nextUpperBound->x != upperBound->x && nextUpperBound->y != upperBound->y)
				{
					upperDiagonalCounter++;
				}
				upperBound = nextUpperBound;
			}
			lowerDiagonalCounter -= 5;
		}

		if (upperDiagonalCounter >= 5)
		{
			for (int i = 0; i < 2 && upperBound > lowerBound; i++)
			{
				nextLowerBound = lowerBound+1;
				if (nextLowerBound->x != lowerBound->x && nextLowerBound->y != lowerBound->y)
				{
					lowerDiagonalCounter++;
				}
				lowerBound = nextLowerBound;
			}
			upperDiagonalCounter -= 5;
		}
	}
	return *lowerBound;
}