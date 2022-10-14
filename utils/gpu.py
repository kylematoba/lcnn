from typing import Any, Tuple
import warnings

import torch


CudaDeviceProperties = Any  # : torch._C._CudaDeviceProperties


def supports_fp16(device_properties: CudaDeviceProperties) -> bool:
    does_not_support_fp16 = ['Tesla K80']
    supports = True
    supports &= (device_properties.name not in does_not_support_fp16)
    return supports


def figure_out_dtype_and_device(want_cuda: bool=True,
                                want_half: bool=True) -> Tuple[torch.dtype, torch.device]:
    use_cuda = torch.cuda.is_available() and want_cuda
    device = torch.device("cuda:0" if use_cuda else "cpu")
    is_gpu = torch.device(type="cpu") != device
    halve = want_half and is_gpu

    if is_gpu:
        current_device = torch.cuda.current_device()
        current_device_properties = torch.cuda.get_device_properties(current_device)
        supports = supports_fp16(current_device_properties)
        if not supports:
            warnings.warn(f"I do not think that this device supports fp16")
        halve = halve and supports
    dtype = torch.float16 if halve else torch.float32
    return dtype, device
