#include "SpectUnet.h"
#include "Helpers.h"

#include <fstream>
#include <numeric>

ml::inference::SpectUnet::SpectUnet(std::filesystem::path model_path)
{
	const char* instance_name = "unet inference";

	env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, instance_name);
	runOptions_ = Ort::RunOptions{ nullptr };

	std::cout << "loading model " << model_path << std::endl;
	session_ = std::make_unique<Ort::Session>(Ort::Session(env_, model_path.c_str(), Ort::SessionOptions{ nullptr }));

	std::unique_ptr<char, Ort::detail::AllocatedFree> temp = session_->GetInputNameAllocated(0, allocator_);
	inputName_ = std::string(temp.get());

	temp = session_->GetOutputNameAllocated(0, allocator_);
	outputName_ = std::string(temp.get());

	Ort::TypeInfo inputInfo = session_->GetInputTypeInfo(0);
	auto inputTensorInfo = inputInfo.GetTensorTypeAndShapeInfo();
	ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
	inputDimensions_ = inputTensorInfo.GetShape();

	Ort::TypeInfo outputInfo = session_->GetOutputTypeInfo(0);
	auto outputTensorInfo = outputInfo.GetTensorTypeAndShapeInfo();
	ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
	outputDimensions_ = outputTensorInfo.GetShape();
}

void ml::inference::SpectUnet::summary()
{
	std::cout << "============= Model Summary =============\n";
	std::cout << "input  name: " << inputName_ << std::endl;
	std::cout << "output name: " << outputName_ << std::endl;
	std::cout << "input  dimensions: " << inputDimensions_ << std::endl;
	std::cout << "output dimensions: " << outputDimensions_ << std::endl;
	std::cout << "=========================================\n";
}

void ml::inference::SpectUnet::infer(const std::string& inpVolFilename, const std::string& outVolFilename,
	const std::filesystem::path& inpVolDir, const std::filesystem::path& outVolDir)
{
	// number of elements per slice
	int64_t nXY = inputDimensions_[2] * inputDimensions_[3];

	// read input volume from binary file
	std::filesystem::path i_path = inpVolDir / inpVolFilename;
	std::ifstream is(i_path, std::ios::in | std::ios::binary);
	if (!is.is_open()) std::cout << "can't open " << i_path << std::endl;
	is.seekg(0, is.end);
	const size_t file_size = is.tellg();
	is.seekg(0, is.beg);
	std::vector<float> inpVol(file_size / sizeof(float));
	is.read(reinterpret_cast<char*>(inpVol.data()), file_size);
	is.close();

	// create output volume
	std::vector<float> outVol(inpVol.size());

	const unsigned int num_slices = inpVol.size() / nXY;

	// input/output tensor values 
	int64_t inpTsorSz = std::accumulate(inputDimensions_.begin(), inputDimensions_.end(), 1, std::multiplies<int64_t>());
	std::vector<float> inpTsorVals(inpTsorSz);
	int64_t outTsorSz = std::accumulate(outputDimensions_.begin(), outputDimensions_.end(), 1, std::multiplies<int64_t>());
	std::vector<float> outTsorVals(outTsorSz);

	// process (2 x slice_stride + 1) at a time
	unsigned int slice_stride = 1;
	for (unsigned int slice_idx = 1; slice_idx < num_slices - 1; ++slice_idx) {

		// fill in input tensor values
		fillInputTensorValues(inpVol, slice_idx, slice_stride, inpTsorVals);

		// create memory info
		Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

		// create input/output tensors
		std::vector<Ort::Value> inpTsors;
		inpTsors.push_back(Ort::Value::CreateTensor<float>(memInfo, inpTsorVals.data(), inpTsorSz, inputDimensions_.data(), inputDimensions_.size()));
		std::vector<Ort::Value> outTsors;
		outTsors.push_back(Ort::Value::CreateTensor<float>(memInfo, outTsorVals.data(), outTsorSz, outputDimensions_.data(), outputDimensions_.size()));

		// run model
		std::unique_ptr<char, Ort::detail::AllocatedFree> itemp = session_->GetInputNameAllocated(0, allocator_);
		std::vector<const char*> inpNames{ itemp.get() };
		std::unique_ptr<char, Ort::detail::AllocatedFree> otemp = session_->GetOutputNameAllocated(0, allocator_);
		std::vector<const char*> outNames{ otemp.get() };
		session_->Run(runOptions_, inpNames.data(), inpTsors.data(), 1, outNames.data(), outTsors.data(), 1);

		// get inference result
		float* outTsorValsPtr = outTsors[0].GetTensorMutableData<float>();
		for (float& val : outTsorVals) {
			val = *outTsorValsPtr;
			outTsorValsPtr++;
		}

		// copy middle slice of output tensor into corresponding slice of output volume
		auto it = std::copy(outTsorVals.begin() + nXY, outTsorVals.begin() + nXY + nXY,
			outVol.begin() + slice_idx * nXY);
		if (it != outVol.begin() + (slice_idx + 1) * nXY) std::perror("copy failed!\n");

	}

	// scale output volume
	float scale = 100'000.f;
	std::for_each(outVol.begin(), outVol.end(), [scale](float& el) {el *= scale; });

	// write output volume to binary file
	std::filesystem::path o_path = outVolDir / outVolFilename;
	std::ofstream os(o_path, std::ios::out | std::ios::binary);
	if (!os.is_open()) std::cout << "can't open " << o_path << std::endl;
	os.write(reinterpret_cast<char*>(outVol.data()), file_size);
	os.close();
}

void ml::inference::SpectUnet::fillInputTensorValues(std::vector<float>& i_vol, unsigned int slice_idx, unsigned int slice_stride,
	std::vector<float>& i_tsor_vals)
{
	int64_t nXY = inputDimensions_[2] * inputDimensions_[3];
	// copy (2 x slice_stride + 1) slices centered at slice_idx to input tensor values
	std::copy(i_vol.begin() + (slice_idx - 1) * nXY,
		i_vol.begin() + (slice_idx + 1) * nXY,
		i_tsor_vals.begin());
}