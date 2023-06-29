#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <unordered_map>

#include "mnist_classifier.h"
/* classes in mnist dataset are digits: 0-9 */
std::vector<std::string> classes = {"0","1","2","3","4","5","6","7","8","9"};

void mapImagesToClasses(std::filesystem::path & i_dir, 
		std::unordered_map<std::string, std::string> & o_map) {
	for (auto label : classes) {
		std::filesystem::path class_dir = i_dir / label;
		int cl_count = 0;
		for (auto const & it : std::filesystem::directory_iterator{class_dir}) {
			o_map[it.path()] = label;
			cl_count++;
		}
		//std::cout << cl_count << " images / class " << label << std::endl;
	} 
}

int main(int argc, char **argv) {
	// Create image classifier
	//std::filesystem::path model_path = "/home/anshu/Downloads";
	//std::string model_name = "mnist-8.onnx";
	//DigitsClassifier ic(model_path / model_name);
	if (argc == 1) {
		std::perror("provide ONNX model file");
		EXIT_FAILURE;
	}
	std::string model_file_name(argv[1]);
	DigitsClassifier ic(model_file_name);

	// Load images in the input directory
	std::filesystem::path imgs_root = "/home/anshu/Downloads/mnist-png/testing";
	std::unordered_map<std::string, std::string> imgs_labels;
	mapImagesToClasses(imgs_root, imgs_labels);
	std::cout << "prepared " << imgs_labels.size() << " test images\n";

	// Inference using image classifier
	std::cout << "\n===== Inference results =====\n";
	int correct = 0, count = 100;
	for (const auto & [img_path, img_label] : imgs_labels) {
		auto y_pred = ic.Inference(img_path);
		//std::cout << "predicted: " << y_pred << ", true: " << img_label << std::endl;
		
		if (img_label == std::to_string(y_pred)) correct++;
		//count--;
		//if (!count) break;
	}
	std::cout << "accuracy: " << (double)correct / imgs_labels.size() * 100.0 << " %\n";
	return 0;
}
