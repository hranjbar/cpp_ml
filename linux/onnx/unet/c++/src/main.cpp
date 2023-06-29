#include <iostream>
#include <filesystem>
#include <iterator>

#include "Unet.h"

int main(int argc, char **argv) {

	if (argc == 1) {
		std::perror("provide ONNX model file");
		EXIT_FAILURE;
	}
	// Initialize model
	std::string model_filename(argv[1]);
	Unet net(model_filename);

	std::cout << "\n === Inference === \n";	// Inference
	std::filesystem::path vols_root_dir = "/home/anshu/horanj/python_C++_deployment/linux/onnx/unet/data/ncRecon";
	for (auto & it : std::filesystem::directory_iterator{vols_root_dir}) {
		std::filesystem::path p = it.path();
		if (p.extension() == ".vol") {
			std::string i_fn = p.filename();
			std::string o_fn = i_fn.substr(0, i_fn.size() - 4) + ".mu";	// done more easily if used 'boost::filesystem::path'
			std::filesystem::path ouput_dir = vols_root_dir / "output";
			std::cout << "\nprocessing " << vols_root_dir / i_fn << std::endl;
			net.Inference(i_fn, o_fn, vols_root_dir, ouput_dir);
		}
	}

	return 0;
}
