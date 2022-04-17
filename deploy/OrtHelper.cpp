#include "OrtHelper.h"
#include <sstream>
#include <unordered_map>

std::string utils::getDataTypeString(ONNXTensorElementDataType type)
{
	static std::unordered_map<ONNXTensorElementDataType, std::string> Map = {
		{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, "UNDEFINED"},
		{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "FLOAT"},
		{ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, "UINT8"},
		{ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, "INT8"},
		{ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, "UINT16"},
		{ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, "INT16"},
		{ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, "INT32"},
		{ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, "INT64"},
		{ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, "STRING"},
		{ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, "BOOL"},
		{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, "FLOAT16"},
		{ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, "DOUBLE"},
		{ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, "UINT32"},
		{ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, "UINT64"},
		{ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64, "COMPLEX64"},
		{ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128, "COMPLEX128"},
		{ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, "BFLOAT16"}
	};
	const auto it = Map.find(type);
	if (it != Map.end())
		return it->second;
	return "UNKNOWN";
}

std::string utils::printShape(const std::vector<int64_t>& v)
{
	std::stringstream ss("");
	for (size_t i = 0; i < v.size() - 1; i++)
		ss << v[i] << "x";
	ss << v[v.size() - 1];
	return ss.str();
}

std::string utils::printShape(const Ort::Value& v)
{
	if (!v.IsTensor())
		return "";
	return printShape(v.GetTensorTypeAndShapeInfo().GetShape());
}

int64_t utils::getShapeElements(const std::vector<int64_t>& v)
{
	int64_t total = 1;
	for (const auto& i : v)
		total *= i;
	return total;
}

Ort::Value utils::createTensor(const std::vector<int64_t>& shape, ONNXTensorElementDataType type)
{
	Ort::AllocatorWithDefaultOptions allocator;
	return Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), type);
}

Ort::Value utils::createTensor(void* p_data, size_t byte_count, const std::vector<int64_t>& shape,
	ONNXTensorElementDataType type)
{
	Ort::MemoryInfo info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	return Ort::Value::CreateTensor(info, p_data, byte_count, shape.data(), shape.size(), type);
}

std::vector<std::string> utils::getAvailableProviderNames()
{
	return Ort::GetAvailableProviders();
}

std::vector<ProviderType> utils::getAvailableProviders()
{
	static std::unordered_map<std::string, ProviderType> Map = {
		{kCpuExecutionProvider, ProviderType::CPU},
		{kCudaExecutionProvider, ProviderType::CUDA},
		{kDnnlExecutionProvider, ProviderType::Dnnl},
		{kOpenVINOExecutionProvider, ProviderType::OpenVINO},
		{kNupharExecutionProvider, ProviderType::Nuphar},
		{kVitisAIExecutionProvider, ProviderType::VitisAI},
		{kTensorrtExecutionProvider, ProviderType::Tensorrt},
		{kNnapiExecutionProvider, ProviderType::Nnapi},
		{kRknpuExecutionProvider, ProviderType::Rknpu},
		{kDmlExecutionProvider, ProviderType::Dml},
		{kMIGraphXExecutionProvider, ProviderType::MIGraphX},
		{kAclExecutionProvider, ProviderType::ACL},
		{kArmNNExecutionProvider, ProviderType::ArmNN},
		{kRocmExecutionProvider, ProviderType::ROCM},
		{kCoreMLExecutionProvider, ProviderType::CoreML},
		{kTvmExecutionProvider, ProviderType::Tvm}
	};
	std::vector<ProviderType> out;
	for (auto&& name : getAvailableProviderNames())
	{
		const auto it = Map.find(name);
		if (it != Map.end())
			out.push_back(it->second);
	}
	return out;
}

void utils::appendCUDAOptions(Ort::SessionOptions& options)
{
	const auto& api = Ort::GetApi();
	OrtCUDAProviderOptionsV2* p = nullptr;
	api.CreateCUDAProviderOptions(&p);
	api.SessionOptionsAppendExecutionProvider_CUDA_V2(options, p);
	api.ReleaseCUDAProviderOptions(p);
}

void utils::appendTensorRTOptions(Ort::SessionOptions& options)
{
	const auto& api = Ort::GetApi();
	OrtTensorRTProviderOptionsV2* p = nullptr;
	api.CreateTensorRTProviderOptions(&p);
	api.SessionOptionsAppendExecutionProvider_TensorRT_V2(options, p);
	api.ReleaseTensorRTProviderOptions(p);
}
