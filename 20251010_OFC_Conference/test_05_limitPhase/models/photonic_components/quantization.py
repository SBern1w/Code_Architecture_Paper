import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# This make the rounding of the forward while the backprop the gradient is linear
class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class DAC_normalized(nn.Module):
    def __init__(self,
                 N_bits=8,
                 V_min=-1.0, V_max=1.0,
                 V_noise_std=0.,):
        super().__init__()
        self.N_bits = N_bits
        self.V_min = V_min
        self.V_max = V_max
        # Quantisation
        self.quantize_ste = QuantizeSTE.apply
        # Noise
        self.V_noise_std = V_noise_std

    def forward(self, Code_normalized):
        # [0, 2pi] -> [0, 2^N_bits-1*2pi] -> quantize -> [-10, 10]
        Code = Code_normalized * (2**self.N_bits-1)
        Code_quantized = self.quantize_ste(Code)
        V_quantized = Code_quantized / (2**self.N_bits-1) * (self.V_max - self.V_min) + self.V_min
        V_noise = V_quantized + torch.randn_like(V_quantized) * self.V_noise_std
        return V_noise
    

class phase_shift_quantization_noise(nn.Module):
    def __init__(self,
                 N_bits=8,
                 phase_shift_noise_std=0.,):
        super().__init__()
        self.N_bits = N_bits
        # Quantisation
        self.quantize_ste = QuantizeSTE.apply
        # Noise
        self.phase_shift_noise_std = phase_shift_noise_std

    def forward(self, phase_shift):
        # phase shift [0, 2pi] -> normalized [0, 1] -> integer [0, 2^N_bits-1]
        phase_shift_integer = phase_shift / (2*np.pi) * (2**self.N_bits-1)
        # quantizied [0, 2^N_bits-1] -> phase shift [0, 2pi]
        phase_shift_quantized = self.quantize_ste(phase_shift_integer) / (2**self.N_bits-1) * (2*np.pi)
        phase_shift_noise = phase_shift_quantized + torch.randn_like(phase_shift_quantized) * self.phase_shift_noise_std
        return phase_shift_noise


