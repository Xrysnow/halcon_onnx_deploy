#pragma once
#include "onnxruntime_cxx_api.h"

class Model
{
public:
	Model(const std::string& path, Ort::Env& env, Ort::SessionOptions& options);
	~Model();

	bool isValid() const { return valid; }

	size_t getInputCount() const { return session.GetInputCount(); }
	std::vector<std::string> getInputNames() const;
	std::vector<std::vector<int64_t>> getInputShapes() const;
	std::vector<ONNXTensorElementDataType> getInputTypes() const;
	size_t getOutputCount() const { return session.GetOutputCount(); }
	std::vector<std::string> getOutputNames() const;
	std::vector<std::vector<int64_t>> getOutputShapes() const;
	std::vector<ONNXTensorElementDataType> getOutputTypes() const;

	std::vector<Ort::Value> run(
		const std::vector<Ort::Value>& input_values,
		const Ort::RunOptions& run_options = {});

private:
	bool valid = false;
	Ort::Session session{ nullptr };
};
