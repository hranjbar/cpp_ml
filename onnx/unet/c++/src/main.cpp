#include <iostream>
#include <filesystem>
#include <unordered_map>

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
	std::cout << "Input file names:\n";
	for (auto const & it : std::filesystem::directory_iterator{vols_root_dir}) {
		if (it.path().extension() == ".vol") {
			std::cout << it.path() << std::endl;
			in_filenames.push_back(it.path());
		}
	}


	// Inference
	std::cout << "\n===== Inference =====\n";
	net.Inference(in_filenames.front());
	//for (const auto & fn : in_filenames) {
		//auto y_pred = net.Inference(fn);
		//std::cout << "predicted: " << y_pred << ", true: " << img_label << std::endl;
	//}
	return 0;
}
