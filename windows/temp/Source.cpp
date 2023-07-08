#include "Tensor.h"
#include "Helpers.h"
#include <array>

int main()
{
	std::filesystem::path fname = "nums.bin";
	/*ml::Tensor<float, 1> t1 = {4.1, .65, -1.3, -.4};
	std::cout << t1 << std::endl;
	t1.writeBin(fname);
	std::cout << "size in bytes: " << t1.bytes() << std::endl;
	ml::Tensor<float, 1> t2(4);
	t2.readBin(fname);
	std::cout << t2 << std::endl;*/

	/*ml::Tensor<float, 3> t3 = { 
		{{3.3, 2.63, -0.12}, {-2.54 ,-0.15, -6.01}}, 
		{{0.3, 0.66, -1.58}, {-0.99 ,-6.07, 9.64}}, 
		{{3.3, 2.63, -0.12}, {-2.54 ,-0.15, -6.01}}, 
		{{0.3, 0.66, -1.58}, {-0.99 ,-6.07, 9.64}},
		{{2.3, -2.63, -1.12}, {2.55 ,0.75, -4.91}}
	};*/
	ml::Tensor<float, 3> t3 = {
		{{3.3f, 2.63f, -0.12f}, {-2.54f ,-0.15f, -6.01f}},
		{{0.3f, 0.66f, -1.58f}, {-0.99f ,-6.07f, 9.64f}},
		{{3.3f, 2.63f, -0.12f}, {-2.54f ,-0.15f, -6.01f}},
		{{0.3f, 0.66f, -1.58f}, {-0.99f ,-6.07f, 9.64f}},
		{{2.3f, -2.63f, -1.12f}, {2.55f ,0.75f, -4.91f}}
	};
	/*ml::Tensor<short, 3> t3 = {
		{{3, 2, -0}, {-2 ,-0, -6}},
		{{0, 0, -1}, {-0 ,-6, 9}},
		{{3, 2, -0}, {-2 ,-0, -6}},
		{{0, 0, -1}, {-0 ,-6, 9}},
		{{2, -2, -1}, {2 ,0, -4}}
	};*/
	//ml::Tensor<float, 3> t3({ 3, 2, 4 });
	std::cout << t3 << std::endl;
	//std::array<std::size_t, 3> dims = { 3, 0, 40000 };
	using namespace ml;
	std::cout << "dimensions: " << t3.dims() << std::endl;
	//std::cout << t3.bytes() << " bytes" << std::endl;
	////std::cout << "1: " << t3[1] << std::endl;
	t3.writeBin(fname);
	ml::Tensor<float, 3> t4({ 3, 2, 5 });
	//ml::Tensor<short, 3> t4({ 3, 2, 5 });
	t4.readBin(fname);
	std::cout << t4 << std::endl;
	std::cout << "dimensions: " << t4.dims() << std::endl;

	
	return 0;
}