#include <iostream>
#include <memory>
#include <filesystem>
#include <string>
#include <chrono>

#include "SpectUnet.h"

void parseInpArgs(std::filesystem::path& model_path, std::filesystem::path& input_data_dir, const int argc, const char** argv);

int main(int argc, char** argv) {

	// parsing inputs
	std::filesystem::path model_path, input_data_dir;
	try {
		parseInpArgs(model_path, input_data_dir, argc, (const char**)argv);
	}
	catch (std::exception& e) {
		std::cout << e.what() << std::endl;
	}

	ml::inference::SpectUnet unet(model_path);
	unet.summary();

	// predict
	for (auto const& entry : std::filesystem::directory_iterator(input_data_dir)) {
		std::filesystem::path p = entry.path();

		// process .vol files only
		if (p.extension() == ".vol") {

			std::string i_fn = p.filename().string();
			std::string o_fn = p.stem().string() + ".mu";

			std::cout << "processing: " << i_fn << std::endl;

			const auto start = std::chrono::system_clock::now();
			unet.infer(i_fn, o_fn, input_data_dir, input_data_dir);
			const auto end = std::chrono::system_clock::now();
			const std::chrono::duration<double> elapsed_seconds = end - start;

			std::cout << elapsed_seconds.count() << " seconds\n" << "output: " << o_fn << std::endl;
		}
	}

	return 0;
}

void parseInpArgs(std::filesystem::path& model_path, std::filesystem::path& input_data_dir, const int argc, const char** argv)
{
	std::string message;
	std::string model_path_arg{ "-m" }, input_data_dir_arg{ "-i" };

	if (argc != 5 || !((argv[1] == model_path_arg && argv[3] == input_data_dir_arg) || (argv[3] == model_path_arg && argv[1] == input_data_dir_arg))) {
		message = "incorrect input arguments!";
		message += "\ninput arguments must be:\n\t" + model_path_arg + " \"path_to_model\" " + input_data_dir_arg + " \"directory_of_input_data\"";
		throw std::exception(message.c_str());
	}

	model_path.clear();
	for (unsigned int i = 1; i < argc; ++i) {
		if (std::string(argv[i]) == model_path_arg) {
			model_path = argv[i + 1];
			if (!std::filesystem::exists(model_path)) {
				message = "model path";
				message += model_path.string() + " doesn't exist!\n";
				throw std::exception(message.c_str());
			}
		}
		else if (std::string(argv[i]) == input_data_dir_arg) {
			input_data_dir = argv[i + 1];
			if (!std::filesystem::exists(input_data_dir)) {
				message = "input data path";
				message += input_data_dir.string() + " doesn't exist!\n";
				throw std::exception(message.c_str());
			}
		}
	}
}