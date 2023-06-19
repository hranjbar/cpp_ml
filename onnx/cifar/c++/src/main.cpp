#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <unordered_map>

#include "image_classifier.h"

/* classes names for CIFAR-10 data-set*/
std::vector<std::string> classes = {"plane", "car",  "bird", "cat",
		"deer",  "dog",  "frog", "horse",
		"ship",  "truck"};

void mapImagesToClasses(std::filesystem::path & i_dir, 
		std::unordered_map<std::string, std::string> & o_map) {
	for (auto label : classes) {
		std::filesystem::path class_dir = i_dir / label;
		for (auto const & it : std::filesystem::directory_iterator{class_dir}) {
			o_map[it.path()] = label;
		}
	} 
}

int main(int argc, char **argv) {
	// Create image classifier
	std::filesystem::path model_path = "/home/anshu/horanj/python_C++_deployment/onnx/pytorch/models";
	std::string model_name = "image_classifier.onnx";
	ImageClassifier ic(model_path / model_name);

	// Load images in the input directory
	std::filesystem::path imgs_root = "/home/anshu/Downloads/cifar10-png/test";
	std::unordered_map<std::string, std::string> imgs_labels;
	mapImagesToClasses(imgs_root, imgs_labels);
	std::cout << "prepared " << imgs_labels.size() << " test images\n";

	// Inference using image classifier
	std::cout << "\n===== Inference results =====\n";
	int correct = 0;
	for (const auto & [img_path, img_label] : imgs_labels) {
		std::string predicted = classes[ic.Inference(img_path)];
		if (img_label == predicted) {
			correct++;
		}
	}
	std::cout << "accuracy: " << (double)correct / imgs_labels.size() * 100.0 << " %\n";
	return 0;
}
