# halcon_onnx_deploy

Deploy ONNX model with Halcon and C++.

## Contents

* `onnx_halcon.hdev`: Use ONNX model in Halcon.
* `onnx_pytorch_convert.py`: Convert a Pytorch model to ONNX model.
* `deploy`: Example to deploy ONNX model with Halcon in C++.

## Requirements

* Halcon 19+ (test on 21.05)
* MSVC 2019+

Python packages to convert Pytorch model:
* torch
* onnx
* onnxruntime
* onnx-simplifier

## Note

ONNX model exported by Pytorch should be processed by `onnx-simplifier`, otherwise it may cause crash in Halcon.
