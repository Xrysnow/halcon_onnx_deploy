#pragma once
#include "halconcpp/HalconCpp.h"
#include <vector>
#include <string>

HalconCpp::HObject ReadImageProcessed(
	const std::string& path,
	size_t targetWidth, size_t targetHeight);

// 'channel' starts from 1
std::vector<float> GetImageFloatData(
	const HalconCpp::HObject& image, int32_t channel = -1);

// classification of image using ONNX model
std::vector<double> ClassifyByDLModel(const HalconCpp::HObject& image,
	const std::string& modelPath, const std::string& outputNodeName);
