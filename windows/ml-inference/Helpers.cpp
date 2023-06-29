#include "Helpers.h"

std::ostream& ML::operator<<(std::ostream& os, const std::vector<int64_t>& v)
{
	os << "[";
	for (int i = 0; i < v.size() - 1; ++i) os << v[i] << ", ";
	os << v.back() << "]";
	return os;
}