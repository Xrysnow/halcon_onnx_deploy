#include "HalconProcess.h"
#include "halconcpp/HalconCpp.h"
#include <iostream>

using namespace HalconCpp;

HObject ReadImageProcessed(
    const std::string& path,
    size_t targetWidth, size_t targetHeight)
{
    HObject  ho_Image, ho_ImageMean, ho_ImageSub;
    HObject  ho_ImageClipped, ho_ImagePreprocessed, ho_ResultObject;
    HTuple  hv_PointerR, hv_PointerG, hv_PointerB, hv_Type, hv_Width, hv_Height;

    ReadImage(&ho_Image, path.c_str());
    ConvertImageType(ho_Image, &ho_Image, "real");
    ReadImage(&ho_ImageMean, "data_mean");
    ConvertImageType(ho_ImageMean, &ho_ImageMean, "real");
    SubImage(ho_Image, ho_ImageMean, &ho_ImageSub, 1.0 / 255, 0.0);
    CropPart(ho_ImageSub, &ho_ImageClipped, 0, 40, 320, 160);

    ZoomImageSize(ho_ImageClipped, &ho_ImagePreprocessed,
        (Hlong)targetWidth, (Hlong)targetHeight, "bilinear");

    return ho_ImagePreprocessed;
}

std::vector<float> GetImageFloatData(const HObject& image, int32_t channel)
{
    HObject ho_Image;
    HTuple hv_Pointer, hv_Type, hv_Width, hv_Height, hv_Channels;
    GetImageSize(image, &hv_Width, &hv_Height);
    GetImageType(image, &hv_Type);
    CountChannels(image, &hv_Channels);
    const bool needConvert = hv_Type != "real";

    if (hv_Channels > 1 && channel >= 1)
    {
        if (hv_Channels < (Hlong)channel)
	    {
            // out of range
            return {};
	    }
        AccessChannel(image, &ho_Image, channel);
        if (needConvert)
        {
            ConvertImageType(ho_Image, &ho_Image, "real");
        }
    }
    else
    {
        if (needConvert)
        {
            ConvertImageType(image, &ho_Image, "real");
        }
        else
        {
			ho_Image = image;
        }
    }

    std::vector<float> out;
	const size_t count = hv_Width.L() * hv_Height.L();
    if (hv_Channels > 1 && channel < 1)
    {
	    // collect all channels
        out.resize(count * hv_Channels.L());
        const auto size = sizeof(float) * count;
        for (Hlong i = 0; i < hv_Channels.L(); ++i)
	    {
            HObject ho_Channel;
            AccessChannel(ho_Image, &ho_Channel, i + 1);
            GetImagePointer1(ho_Channel, &hv_Pointer, &hv_Type, &hv_Width, &hv_Height);
            const auto pointer = (float*)hv_Pointer.L();
            memcpy(out.data() + count * i, pointer, size);
	    }
    }
    else
    {
	    // one channel
        out.resize(count);
        const auto size = sizeof(float) * count;
        GetImagePointer1(ho_Image, &hv_Pointer, &hv_Type, &hv_Width, &hv_Height);
        const auto pointer = (float*)hv_Pointer.L();
        memcpy(out.data(), pointer, size);
    }
    return out;
}

std::vector<double> ClassifyByDLModel(const HObject& image,
    const std::string& modelPath, const std::string& outputNodeName)
{
    HObject ho_ImagePreprocessed, ho_ResultObject;
    HObject ho_Channel;

    HTuple hv_ClassNames, hv_DLModelHandle, hv_Info, hv_NumGPU;
    HTuple hv_ProcessImageDim;
    HTuple hv_DLSample, hv_DLResult;
    HTuple hv_ResultLength, hv_ChannelValue;

    HTuple hv_ModelPath = modelPath.c_str();
    HTuple hv_OutputNodeName = outputNodeName.c_str();

    //load onnx model
    ReadDlModel(hv_ModelPath, &hv_DLModelHandle);
    //set model
    GetSystem("cuda_devices", &hv_Info);
    TupleLength(hv_Info, &hv_NumGPU);
    if (0 != (int(hv_NumGPU == 0)))
    {
        SetDlModelParam(hv_DLModelHandle, "runtime", "cpu");
    }
    SetDlModelParam(hv_DLModelHandle, "batch_size", 1);
    SetDlModelParam(hv_DLModelHandle, "type", "generic");
    SetDlModelParam(hv_DLModelHandle, "runtime_init", "immediately");
    //get input size from model
    GetDlModelParam(hv_DLModelHandle, "input_dimensions", &hv_ProcessImageDim);
    HTuple hv_InputKey = "input";
    GetDictTuple(hv_ProcessImageDim, hv_InputKey, &hv_ProcessImageDim);

    //scale to input size
    ZoomImageSize(image, &ho_ImagePreprocessed,
        hv_ProcessImageDim[0], hv_ProcessImageDim[1], "bilinear");

    //prepare input
    CreateDict(&hv_DLSample);
    SetDictObject(ho_ImagePreprocessed, hv_DLSample, hv_InputKey);
    //compute
    ApplyDlModel(hv_DLModelHandle, hv_DLSample, HTuple(), &hv_DLResult);
    //get result (will be multi-channel image if type is 'generic')
    GetDictObject(&ho_ResultObject, hv_DLResult, hv_OutputNodeName);
    CountChannels(ho_ResultObject, &hv_ResultLength);
    std::vector<double> out;
    for (HTuple hv_Index = 1; hv_Index.Continue(hv_ResultLength, 1); hv_Index += 1)
    {
	    AccessChannel(ho_ResultObject, &ho_Channel, hv_Index);
	    GetGrayval(ho_Channel, 0, 0, &hv_ChannelValue);
	    out.push_back(hv_ChannelValue.D());
    }
    return out;
}
