#include <iostream>
#include <filesystem>
#include <unordered_map>
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

	// populate input file names
	std::filesystem::path vols_root_dir = "/home/anshu/horanj/python_C++_deployment/onnx/unet/data/ncRecon";
	std::vector<std::string> in_filenames;
	for (auto const & it : std::filesystem::directory_iterator{vols_root_dir}) {
		if (it.path().extension() == ".vol") {
			in_filenames.push_back(it.path());
		}
	}

	// Inference
	std::cout << "\n===== Inference =====\n";
	for (const auto & fn : in_filenames) {
		std::cout << "volume: " << fn << std::endl;
		net.Inference(fn, fn + ".output");
	}
	return 0;
}
