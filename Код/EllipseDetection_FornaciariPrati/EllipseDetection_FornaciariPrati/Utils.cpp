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

 // сохраняем правильно распознанные на данном изображении эллипсы в базу
 // вводим индексы эллипсов, конец ввода - Ctrl+Z или не-число
 void saveEllipses(const std::vector<Ellipse>& ellipses, const std::string& imgName, const Size& imgSize)
 {
	 int i;
	 std::ofstream out("C:\\Диплом\\Images\\калибровка\\db.txt", std::ios_base::app);
	 out << imgName << " " << imgSize.width << " " << imgSize.height << std::endl;
	 std::vector<int> indexes;
	 while (std::cin >> i)
	 {
		 indexes.push_back(i);
	 }
	 out << indexes.size() << std::endl;
	 for (auto idx : indexes)
		 out << ellipses[idx];
	 out << std::endl;
	 out.close();
 }

void loadEllipses(std::map<std::string, std::pair<Size, std::vector<Ellipse>>>& ellipses)
{
	std::ifstream in("C:\\Диплом\\Images\\калибровка\\db.txt");
	std::string imgName;
	int numberOfEllipses;
	Ellipse e;

	while (in >> imgName)
	{
		auto& thisImgEllipses = ellipses[imgName];
		in >> thisImgEllipses.first.width >> thisImgEllipses.first.height;
		in >> numberOfEllipses;
		for (int i = 0; i < numberOfEllipses; i++)
		{
			in >> e;
			e.SetImgSize(thisImgEllipses.first);
			thisImgEllipses.second.push_back(e);
		}
	}

}