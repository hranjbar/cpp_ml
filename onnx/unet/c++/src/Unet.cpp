#include "Unet.h"

#include <algorithm>

#ifdef VERBOSE
/**
 * @brief Print ONNX tensor type
 * https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L93
 * @param os
 * @param type
 * @return std::ostream&
 */
std::ostream& operator<<(std::ostream& os,
                         const ONNXTensorElementDataType& type) {
  switch (type) {
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
      os << "undefined";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      os << "float";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      os << "uint8_t";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      os << "int8_t";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      os << "uint16_t";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      os << "int16_t";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      os << "int32_t";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      os << "int64_t";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      os << "std::string";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      os << "bool";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      os << "float16";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      os << "double";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      os << "uint32_t";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      os << "uint64_t";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      os << "float real + float imaginary";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      os << "double real + float imaginary";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      os << "bfloat16";
      break;
    default:
      break;
  }

  return os;
}
#endif

// Constructor
Unet::Unet(const std::string& modelFilepath) {
  /**************** Create ORT environment ******************/
  std::string instanceName{"UNet inference"};
  mEnv = std::make_shared<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                    instanceName.c_str());

  /**************** Create ORT session ******************/
  // Set up options for session
  Ort::SessionOptions sessionOptions;
  // Enable CUDA
  sessionOptions.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
  // Sets graph optimization level (Here, enable all possible optimizations)
  sessionOptions.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  // Create session by loading the onnx model
  mSession = std::make_shared<Ort::Session>(*mEnv, modelFilepath.c_str(),
    sessionOptions);

  /**************** Create allocator ******************/
  // Allocator is used to get model information
  Ort::AllocatorWithDefaultOptions allocator;

  /**************** Input info ******************/
  // Get the number of input nodes
  size_t numInputNodes = mSession->GetInputCount();
#ifdef VERBOSE
  std::cout << "******* Model information below *******" << std::endl;
  std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
#endif

  // Get the name of the input
  // 0 means the first input of the model
  // The example only has one input, so use 0 here
  std::unique_ptr<char, Ort::detail::AllocatedFree> temp = mSession->GetInputNameAllocated(0, allocator);
  mInputName = temp.get();
#ifdef VERBOSE
  std::cout << "Input Name: " << std::string(mInputName) << std::endl;
#endif

  // Get the type of the input
  // 0 means the first input of the model
  Ort::TypeInfo inputTypeInfo = mSession->GetInputTypeInfo(0);
  auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
#ifdef VERBOSE
  std::cout << "Input Type: " << inputType << std::endl;
#endif

  // Get the shape of the input
  mInputDims = inputTensorInfo.GetShape();
#ifdef VERBOSE
  std::cout << "Input Dimensions: " << mInputDims << std::endl;
#endif

  /**************** Output info ******************/
  // Get the number of output nodes
  size_t numOutputNodes = mSession->GetOutputCount();
#ifdef VERBOSE
  std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;
#endif

  // Get the name of the output
  // 0 means the first output of the model
  // The example only has one output, so use 0 here
  temp = mSession->GetOutputNameAllocated(0, allocator);
  mOutputName = temp.get();
#ifdef VERBOSE
  std::cout << "Output Name: " << std::string(mOutputName) << std::endl;
#endif

  // Get the type of the output
  // 0 means the first output of the model
  Ort::TypeInfo outputTypeInfo = mSession->GetOutputTypeInfo(0);
  auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
#ifdef VERBOSE
  std::cout << "Output Type: " << outputType << std::endl;
#endif

  // Get the shape of the output
  mOutputDims = outputTensorInfo.GetShape();
#ifdef VERBOSE
  std::cout << "Output Dimensions: " << mOutputDims << std::endl << std::endl;
#endif
}


/* Performs inference */
void Unet::Inference(const std::string & inputVolumeFilename, const std::string & outputVolumeFilename, 
    const std::filesystem::path & inputVolumeDirectory, const std::filesystem::path & outputVolumeDirectory)
{

  // read input volume from binary file
  std::filesystem::path i_path = inputVolumeDirectory / inputVolumeFilename;
  std::ifstream is(i_path, std::ios::in | std::ios::binary);
  if (!is.is_open()) std::cout << "can't open: " << i_path << " for reading!\n";
  is.seekg(0, is.end);
  const size_t filesize = is.tellg();
  is.seekg(0, is.beg);
  std::vector<float> inputVolume(filesize / sizeof(float));
  is.read(reinterpret_cast<char *>(inputVolume.data()), filesize);
  is.close();

  // allocate output volume 
  std::vector<float> outputVolume(inputVolume.size());

  // calculate size of volumes
  int64_t nXY = mInputDims[2] * mInputDims[3];
  const unsigned int numSlices = inputVolume.size() / nXY;

  // create input/output tensor size
  int64_t inputTensorSize = std::accumulate(mInputDims.begin(), mInputDims.end(), 1, std::multiplies<int64_t>());
  std::vector<float> inputTensorValues(inputTensorSize);
  int64_t outputTensorSize = std::accumulate(mOutputDims.begin(), mOutputDims.end(), 1, std::multiplies<int64_t>());
  std::vector<float> outputTensorValues(outputTensorSize);

  NormalizeArray(inputVolume);
  // slc_ix = [1 , ... , n-1], processing 3 adjacent slices at a time, moving 1 slice each iteration
  for (unsigned int slc_ix = 1; slc_ix < numSlices - 1; slc_ix++) {
    // create allocator
    Ort::AllocatorWithDefaultOptions allocator;

    // prepare input tensor values
    PopulateInputTensorValues(inputVolume, slc_ix, inputTensorValues);
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, 
      OrtMemType::OrtMemTypeDefault);

    // create input tensors
    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
      memoryInfo, inputTensorValues.data(), inputTensorSize, mInputDims.data(),
      mInputDims.size()));

    // create output tensors
    std::vector<Ort::Value> outputTensors;
    outputTensors.push_back(Ort::Value::CreateTensor<float>(
      memoryInfo, outputTensorValues.data(), outputTensorSize,
      mOutputDims.data(), mOutputDims.size()));

    // inference: input tensor --> output tensor
    std::unique_ptr<char, Ort::detail::AllocatedFree> itemp = mSession->GetInputNameAllocated(0, allocator);
    std::vector<const char*> inputNames{itemp.get()};
    std::unique_ptr<char, Ort::detail::AllocatedFree> otemp = mSession->GetOutputNameAllocated(0, allocator);
    std::vector<const char*> outputNames{otemp.get()};
    mSession->Run(Ort::RunOptions{nullptr}, inputNames.data(),
      inputTensors.data(), 1, outputNames.data(), outputTensors.data(), 1);

    // Get the inference result
    float * outputTensorValuesPtr = outputTensors.front().GetTensorMutableData<float>();
    for (float & val : outputTensorValues) {
      val = *outputTensorValuesPtr;
      outputTensorValuesPtr++;
    }

    // copy the middle slice of 3 slices in outputTensorValues to slice slc_ix of outputVolume
    auto it = std::copy(outputTensorValues.begin() + nXY, outputTensorValues.begin() + (nXY + nXY), 
      outputVolume.begin() + slc_ix * nXY);
    if (it != outputVolume.begin() + (slc_ix + 1) * nXY) std::perror("copy failed!\n");

    // std::cout << "slice: " << slc_ix + 1 << std::endl;

  } // slc_ix

  // scale output volume
  float scale = 100'000;
  std::for_each(outputVolume.begin(), outputVolume.end(), [scale](float & el){el *= scale;});

  // write output volume to binary file
  std::filesystem::path o_path = outputVolumeDirectory / outputVolumeFilename;
  std::ofstream os(o_path, std::ios::out | std::ios::binary);
  if (!os.is_open()) std::cout << "can't open " << o_path << " for writing!\n";
  else std::cout << "output mu-map: " << o_path << std::endl;
  os.write(reinterpret_cast<char *>(outputVolume.data()), filesize);
  os.close();

  /**************** Preprocessing ******************/
  // Create input tensor (including size and value) from the loaded input image
// #ifdef TIME_PROFILE
//   const auto before = clock_time::now();
// #endif

// #ifdef TIME_PROFILE
//   const sec duration = clock_time::now() - before;
//   std::cout << "The preprocessing takes " << duration.count() << "s"
//             << std::endl;
// #endif
}

/* divide all elements by maximum */
void Unet::NormalizeArray(std::vector<float> & inputArray)
{
  float const maxValue = *std::max_element(inputArray.begin(), inputArray.end());
  if (maxValue > std::numeric_limits<float>::epsilon()) {
    std::transform(inputArray.begin(), inputArray.end(), inputArray.begin(),
    [&maxValue](float & el){
      return el / maxValue;
    });
  } else std::perror("could not normalize array, maximum value too small!\n");
}

/* create input tensor values */
void Unet::PopulateInputTensorValues(std::vector<float> & inputVolume, 
  unsigned int sliceIndex, std::vector<float>& inputTensorValues)
{
  int64_t nXY = mInputDims[2] * mInputDims[3];
  int st = (sliceIndex - 1) * nXY;
  int en = (sliceIndex + 1) * nXY;
  std::copy(inputVolume.begin() + st, inputVolume.begin() + en, inputTensorValues.begin());
}