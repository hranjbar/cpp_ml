#include <iostream>
#include <memory>
#include <filesystem>
#include <string>
#include <chrono>

#include "SpectUnet.h"

void parseInpArgs(std::filesystem::path& model_path, const int argc, const char** argv);

int main(int argc, char** argv) {

	// input model 
	std::filesystem::path model_path;
	parseInpArgs(model_path, argc, (const char**)argv);
	std::filesystem::path data_dir("G:\\inference_models_data\\data");
	std::string model_filename = "unet.onnx";

	ml::runtime::SpectUnet unet(model_path);
	unet.summary();

	// predict
	for (auto const& entry : std::filesystem::directory_iterator(data_dir)) {
		std::filesystem::path p = entry.path();

		// process .vol files only
		if (p.extension() == ".vol") {

			std::string i_fn = p.filename().string();
			std::string o_fn = p.stem().string() + ".mu";

			std::cout << "processing: " << i_fn << std::endl;

			const auto start = std::chrono::system_clock::now();
			unet.infer(i_fn, o_fn, data_dir, data_dir);
			const auto end = std::chrono::system_clock::now();
			const std::chrono::duration<double> elapsed_seconds = end - start;

			std::cout << elapsed_seconds.count() << " seconds\n" << "output: " << o_fn << std::endl;
		}
	}

	return 0;
}

void parseInpArgs(std::filesystem::path& model_path, const int argc, const char** argv)
{
	std::string model_path_arg{ "-m" };
	model_path.clear();
	for (unsigned int i = 1; i < argc; ++i) {
		if (std::string(argv[i]) == model_path_arg) {
			model_path = argv[i + 1];
			if (!std::filesystem::exists(model_path)) {
				std::string message{ "model path " };
				message += model_path.string() + " doesn't exist!\n";
				throw std::exception(message.c_str());
			}
		}
	}
}