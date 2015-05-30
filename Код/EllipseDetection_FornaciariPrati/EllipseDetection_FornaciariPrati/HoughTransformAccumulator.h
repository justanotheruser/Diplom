#include <vector>

class HoughTransformAccumulator
{
public:
	void Add(int value);
	int FindMax();
private:
	std::vector<int> m_points;
};

