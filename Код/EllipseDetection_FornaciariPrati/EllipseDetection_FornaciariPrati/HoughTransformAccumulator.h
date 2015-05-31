#include <map>

class HoughTransformAccumulator
{
public:
	void Add(double value);
	double FindMax();
private:
	std::map<double, int> m_points;
};

