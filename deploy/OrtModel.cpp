#include "OrtModel.h"
#include <unordered_map>

Model::Model(const std::string& path, Ort::Env& env, Ort::SessionOptions& options)
{
#ifdef _WIN32
	const std::wstring wide_string = std::wstring(path.begin(), path.end());
	const auto model_file = std::basic_string<ORTCHAR_T>(wide_string);
#else
	const std::string model_file = path;
#endif
	try
	{
		session = Ort::Session(env, model_file.c_str(), options);
		valid = true;
	}
	catch (...)
	{
	}
}

Model::~Model()
{
}

std::vector<std::string> Model::getInputNames() const
{
	Ort::AllocatorWithDefaultOptions allocator;
	const size_t node_count = session.GetInputCount();
	std::vector<std::string> out(node_count);
	for (size_t i = 0; i < node_count; i++) {
		char* tmp = session.GetInputName(i, allocator);
		out[i] = tmp;
		allocator.Free(tmp);  // prevent memory leak
	}
	return out;
}

std::vector<std::vector<int64_t>> Model::getInputShapes() const
{
	const size_t node_count = session.GetInputCount();
	std::vector<std::vector<int64_t>> out(node_count);
	for (size_t i = 0; i < node_count; i++)
		out[i] = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
	return out;
}

std::vector<ONNXTensorElementDataType> Model::getInputTypes() const
{
	const size_t node_count = session.GetInputCount();
	std::vector<ONNXTensorElementDataType> out(node_count);
	for (size_t i = 0; i < node_count; i++)
		out[i] = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
	return out;
}

std::vector<std::string> Model::getOutputNames() const
{
	Ort::AllocatorWithDefaultOptions allocator;
	const size_t node_count = session.GetOutputCount();
	std::vector<std::string> out(node_count);
	for (size_t i = 0; i < node_count; i++) {
		char* tmp = session.GetOutputName(i, allocator);
		out[i] = tmp;
		allocator.Free(tmp);  // prevent memory leak
	}
	return out;
}

std::vector<std::vector<int64_t>> Model::getOutputShapes() const
{
	const size_t node_count = session.GetOutputCount();
	std::vector<std::vector<int64_t>> out(node_count);
	for (size_t i = 0; i < node_count; i++)
		out[i] = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
	return out;
}

std::vector<ONNXTensorElementDataType> Model::getOutputTypes() const
{
	const size_t node_count = session.GetOutputCount();
	std::vector<ONNXTensorElementDataType> out(node_count);
	for (size_t i = 0; i < node_count; i++)
		out[i] = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
	return out;
}

std::vector<Ort::Value> Model::run(
	const std::vector<Ort::Value>& input_values,
	const Ort::RunOptions& run_options)
{
	const size_t input_count = session.GetInputCount();
	const size_t output_count = session.GetOutputCount();
	std::vector<Ort::Value> output_values;
	for (size_t i = 0; i < output_count; i++)
		output_values.emplace_back(nullptr);
	const auto input_names = getInputNames();
	const auto output_names = getOutputNames();
	std::vector<const char*> input_names_(input_count, nullptr);
	size_t i = 0;
	for (const auto& input_name : input_names)
		input_names_[i++] = input_name.c_str();
	std::vector<const char*> output_names_(output_count, nullptr);
	i = 0;
	for (const auto& output_name : output_names)
		output_names_[i++] = output_name.c_str();
	session.Run(run_options,
		input_names_.data(), input_values.data(), input_count,
		output_names_.data(), output_values.data(), output_count);
	return output_values;
}
