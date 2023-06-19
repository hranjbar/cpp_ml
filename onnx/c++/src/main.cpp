#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <unordered_map>

#include "dirent.h"
#include "image_classifier.h"

/* classes names for CIFAR-10 data-set*/
// Classes
  std::vector<std::string> classes = {"plane", "car",  "bird", "cat",
                                      "deer",  "dog",  "frog", "horse",
                                      "ship",  "truck"};

/**
 * @brief Get all the image filenames in a specified directory
 * @param img_dir: the input directory
 * @param img_names: the vector storing all the image filenames
 */
void getAllImageFiles(const std::string &img_dir,
                      std::vector<std::string> &img_names) {
  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir(img_dir.c_str())) != NULL) {
    while ((ent = readdir(dir)) != NULL) {
      std::string filename(ent->d_name);
      if (filename == "." || filename == "..") continue;
      img_names.push_back(filename);
    }
    closedir(dir);
  } else {
    // Failed to open directory
    perror("");
    exit(EXIT_FAILURE);
  }
}

void mapImagesToClasses(std::filesystem::path & i_dir, 
		std::unordered_map<std::string, std::string> & o_map) {
	for (auto label : classes) {
		std::filesystem::path class_dir = i_dir / label;
		int ct = 0;
		for (auto const & it : std::filesystem::directory_iterator{class_dir}) {
			//std::cout << it.path() << std::endl;
			const std::string & fname = it.path();
			o_map[fname] = label;
			ct++;
		}
		std::cout << ct << " images for " << label << " class\n";
	} 
}

int main(int argc, char **argv) {
  // Create image classifier
  ImageClassifier ic("../../pytorch/models/image_classifier.onnx");

  // Load images in the input directory
  std::filesystem::path imgs_root = "/home/anshu/Downloads/cifar10-png/test";
  std::unordered_map<std::string, std::string> imgs_labels;
  mapImagesToClasses(imgs_root, imgs_labels);
  std::cout << "Successfully prepared " << imgs_labels.size() << " test images\n";
  /*std::string img_dir("../../CIFAR_images/");
  std::vector<std::string> img_names;
  getAllImageFiles(img_dir, img_names);
  std::cout << "Following images were loaded: \t";
  for (auto fn : img_names) std::cout << fn << " ";
  std::cout << std::endl << std::endl;*/

  // Inference using image classifier
  /*std::cout << "******* Predicition results below *******" << std::endl;
  for (int i = 0; i < int(img_names.size()); ++i) {
    std::string img_path = img_dir + img_names[i];
    std::cout << "Loaded image: " << img_path << std::endl;
    int cls_idx = ic.Inference(img_path);
    std::cout << "Predicted class: " << classes[cls_idx] << std::endl
              << std::endl;
  }
  std::cout << "Successfully performed image classification" << std::endl;*/
  int correct = 0;
  for (const auto & x : imgs_labels) {
	  const std::string & img_path = x.first;
	  const std::string & img_label = x.second;
	  std::string predicted = classes[ic.Inference(img_path)];
	  if (img_label == predicted) {
		  correct++;
		  //std::cout << "predicted " << img_path << " as " << predicted << std::endl;
	  }
  }
  std::cout << "accuracy: " << (double)correct / imgs_labels.size() << std::endl;
  return 0;
}
