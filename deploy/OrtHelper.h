#pragma once
#include "onnxruntime_cxx_api.h"
#include <cmath>
#include <algorithm>

enum class ProviderType
{
	CPU,
	CUDA,
	Dnnl,
	OpenVINO,
	Nuphar,
	VitisAI,
	Tensorrt,
	Nnapi,
	Rknpu,
	Dml,
	MIGraphX,
	ACL,
	ArmNN,
	ROCM,
	CoreML,
	Tvm,
};

namespace utils
{
	std::string getDataTypeString(ONNXTensorElementDataType type);
	std::string printShape(const std::vector<int64_t>& v);
	std::string printShape(const Ort::Value& v);
	int64_t getShapeElements(const std::vector<int64_t>& v);

	template <typename T>
	Ort::Value createTensor(const std::vector<int64_t>& shape) {
		return createTensor(shape, Ort::TypeToTensorType<T>::type);
	}
	Ort::Value createTensor(const std::vector<int64_t>& shape, ONNXTensorElementDataType type);
	template <typename T>
	Ort::Value createTensor(T* p_data,
		const std::vector<int64_t>& shape) {
		return createTensor(p_data, getShapeElements(shape) * sizeof(T), shape, Ort::TypeToTensorType<T>::type);
	}
	Ort::Value createTensor(void* p_data, size_t byte_count,
		const std::vector<int64_t>& shape, ONNXTensorElementDataType type);

	template <typename T>
	std::vector<float> softmax(const T& input) {
		float rowmax = *std::max_element(input.begin(), input.end());
		std::vector<float> y(input.size());
		float sum = 0.0f;
		for (size_t i = 0; i != input.size(); ++i)
			sum += y[i] = std::exp(input[i] - rowmax);
		for (size_t i = 0; i != input.size(); ++i)
			y[i] = y[i] / sum;
		return y;
	}

	std::vector<std::string> getAvailableProviderNames();
	std::vector<ProviderType> getAvailableProviders();
	void appendCUDAOptions(Ort::SessionOptions& options);
	void appendTensorRTOptions(Ort::SessionOptions& options);

	constexpr const char* kCpuExecutionProvider = "CPUExecutionProvider";
	constexpr const char* kCudaExecutionProvider = "CUDAExecutionProvider";
	constexpr const char* kDnnlExecutionProvider = "DnnlExecutionProvider";
	constexpr const char* kOpenVINOExecutionProvider = "OpenVINOExecutionProvider";
	constexpr const char* kNupharExecutionProvider = "NupharExecutionProvider";
	constexpr const char* kVitisAIExecutionProvider = "VitisAIExecutionProvider";
	constexpr const char* kTensorrtExecutionProvider = "TensorrtExecutionProvider";
	constexpr const char* kNnapiExecutionProvider = "NnapiExecutionProvider";
	constexpr const char* kRknpuExecutionProvider = "RknpuExecutionProvider";
	constexpr const char* kDmlExecutionProvider = "DmlExecutionProvider";
	constexpr const char* kMIGraphXExecutionProvider = "MIGraphXExecutionProvider";
	constexpr const char* kAclExecutionProvider = "ACLExecutionProvider";
	constexpr const char* kArmNNExecutionProvider = "ArmNNExecutionProvider";
	constexpr const char* kRocmExecutionProvider = "ROCMExecutionProvider";
	constexpr const char* kCoreMLExecutionProvider = "CoreMLExecutionProvider";
	constexpr const char* kTvmExecutionProvider = "TvmExecutionProvider";
}
