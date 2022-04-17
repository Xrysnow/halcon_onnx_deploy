import os
import numpy as np
import torch
import torch.onnx
import onnx
import onnxruntime
import onnxsim

def convert_saved(net, input_size, src_path, dst_path, version=10, dynamic_batch=False):
    net.to(torch.device("cpu"))
    net.load_state_dict(torch.load(src_path))
    net.eval()
    # fake input
    x = torch.randn(*input_size, requires_grad=True)
    torch_out = net(x)
    dynamic_axes = None
    if dynamic_batch:
        # variable lenght axes
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    torch.onnx.export(net,
                      x,  # model input (or a tuple for multiple inputs)
                      dst_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=version,  # the ONNX version to export the model to
                      do_constant_folding=False,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      keep_initializers_as_inputs=False,
                      dynamic_axes=dynamic_axes
                      )

    onnx_model = onnx.load(dst_path)
    onnx.checker.check_model(onnx_model)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ort_session = onnxruntime.InferenceSession(dst_path)
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    # onnx simplifier
    os.system('onnxsim %s %s' % (dst_path, dst_path))
