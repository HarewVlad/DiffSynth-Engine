import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager


def enable_fp8_linear(module: nn.Module):
    _enable_fp8_linear(module)
    setattr(module, "fp8_linear_enabled", True)


def _enable_fp8_linear(module: nn.Module):
    if isinstance(module, nn.Linear) and torch.is_floating_point(module.weight.data):
        # avoid conversion for int weights like GGUF
        module.weight.data = module.weight.data.to(torch.float8_e4m3fn)
    for submodule in module.children():
        _enable_fp8_linear(submodule)


@contextmanager
def fp8_inference(enabled=True):
    if not enabled:
        yield
        return

    origin_linear = F.linear

    def fp8_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = input.device
        origin_dtype = input.dtype
        input = input.to(torch.float8_e4m3fn)
        weight = weight.to(torch.float8_e4m3fn)

        if len(input.shape) > 2:
            origin_shape = input.shape
            input = input.reshape(-1, origin_shape[-1])
            scaled_mm_result = torch._scaled_mm(
                input,
                weight.T,
                scale_a=torch.tensor(1.0).to(device=device),
                scale_b=torch.tensor(1.0).to(device=device),
                bias=bias,
                out_dtype=origin_dtype,
            )

            # Extract the actual tensor from the tuple
            if isinstance(scaled_mm_result, tuple):
                result = scaled_mm_result[0]
            else:
                result = scaled_mm_result

            new_shape = origin_shape[:-1] + result.shape[-1:]
            result = result.reshape(new_shape)
        else:
            scaled_mm_result = torch._scaled_mm(
                input,
                weight.T,
                scale_a=torch.tensor(1.0).to(device=device),
                scale_b=torch.tensor(1.0).to(device=device),
                bias=bias,
                out_dtype=origin_dtype,
            )

            # Extract the actual tensor from the tuple
            if isinstance(scaled_mm_result, tuple):
                result = scaled_mm_result[0]
            else:
                result = scaled_mm_result

        return result

    F.linear = fp8_linear
    yield
    F.linear = origin_linear
