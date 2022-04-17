#include "HalconProcess.h"
#include "halconcpp/HalconCpp.h"
#include "onnxruntime_cxx_api.h"
#include "OrtModel.h"
#include "OrtHelper.h"
#include "HalconProcess.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <sstream>
#include <vector>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

static bool run(Model& model, const std::string& path)
{
    std::vector<float> input_values;
    try
    {
        input_values = GetImageFloatData(ReadImageProcessed(path, 80, 80));
    }
    catch (HalconCpp::HException& e)
    {
        cout << "ERROR running Halcon: " << e.ErrorMessage() << endl;
        return false;
    }
    catch (...)
    {
        cout << "ERROR running Halcon" << endl;
        return false;
    }
    const auto input_shape = model.getInputShapes()[0];
    if (input_values.size() != utils::getShapeElements(input_shape))
    {
        cout << "Invalid data size: " << input_values.size() << endl;
        return false;
    }

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(
        utils::createTensor<float>(input_values.data(), input_shape));
    try
    {
	    const auto output_tensors = model.run(input_tensors);
        const auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        //cout << "Output shape: " << utils::printShape(output_shape) << endl;

    	cout << std::setw(10) << path << " ";
        const auto pointer = output_tensors[0].GetTensorData<float>();
        if (pointer[0] > pointer[1])
            cout << "ok ";
        else
            cout << "ng ";
        std::vector<float> out;
        out.reserve(2);
        for (int i = 0; i < 2; ++i)
	        out.push_back(pointer[i]);
        out = utils::softmax(out);
    	cout << std::setw(8) << std::fixed << *std::max_element(out.begin(), out.end());
        //cout << " (" << pointer[0] << " " << pointer[1] << ")";
        cout << endl;
    }
    catch (const Ort::Exception& exception)
    {
        cout << "ERROR running model: " << exception.what() << endl;
        return false;
    }
    return true;
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        cout << "Usage: ./<exe> <model.onnx>" << endl;
        return -1;
    }

    cout << "Available providers:" << endl;
    for (auto&& name : utils::getAvailableProviderNames())
        cout << " " << name << " ";
    cout << endl;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;

    Model model(argv[1], env, session_options);

    for (auto&& p : fs::directory_iterator("ok"))
	    run(model, p.path().string());
    for (auto&& p : fs::directory_iterator("ng"))
	    run(model, p.path().string());
	return 0;
}
