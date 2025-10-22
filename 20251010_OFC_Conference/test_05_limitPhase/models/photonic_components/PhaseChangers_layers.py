import torch
import torch.nn as nn

from models.photonic_components.quantization import *

# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# class HeaterLayerMatrix_Full(nn.Module):
#     r"""
#     Create waveguide matrix layer with all lines connected to a ht
#         0__[h]__0

#         1__[h]__1

#         2__[h]__2
        
#         3__[h]__3
#     """
#     def __init__(self,
#                  n_inputs: int):
#         super(HeaterLayerMatrix_Full, self).__init__()
#         if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
#         self.phase_shift = nn.Parameter(2*torch.pi*torch.rand(n_inputs, device=device), requires_grad=True)

#     def forward(self, x_matrix):
#         layer = torch.exp(1.j*self.phase_shift)
#         return layer.view(-1, 1) * x_matrix



# class HeaterLayerMatrix_Even(nn.Module):
#     r"""
#     Create waveguide matrix layer with just even lines connected to a ht
#         0__[h]__0

#         1_______1

#         2__[h]__2
        
#         3_______3
#     """
#     def __init__(self,
#                  n_inputs: int,):
#         super(HeaterLayerMatrix_Even, self).__init__()
#         if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
#         self.phase_shift = nn.Parameter(2*torch.pi*torch.rand(n_inputs, device=device), requires_grad=True)
#         self.conn = torch.ones((n_inputs), device=device)
#         self.mask = torch.tensor([i % 2 == 0 for i in range(n_inputs)], dtype=torch.bool)

#     def forward(self, x_matrix):
#         layer = torch.where(self.mask, torch.exp(1.j*self.phase_shift), self.conn)
#         return layer.view(-1, 1) * x_matrix



# class HeaterLayerMatrix_Odd(nn.Module):
#     r""" Create waveguide matrix layer with just odd lines connected to a ht
#     0_______0

#     1__[h]__1

#     2_______2
    
#     3_______3
#     """
#     def __init__(self,
#                  n_inputs: int,):
#         super(HeaterLayerMatrix_Odd, self).__init__()
#         if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
#         self.phase_shift = nn.Parameter(2*torch.pi*torch.rand(n_inputs, device=device), requires_grad=True)
#         self.mask = torch.tensor([i % 2 == 1 for i in range(n_inputs)], dtype=torch.bool)
#         self.mask[-1] = False
#         self.conn = torch.ones((n_inputs), device=device)

#     def forward(self, x_matrix):
#         layer = torch.where(self.mask, torch.exp(1.j*self.phase_shift), self.conn)
#         return layer.view(-1, 1) * x_matrix


# class HeaterLayerMatrix_Side(nn.Module):
#     r""" Create waveguide matrix layer with just the top and bottom lines connected to a ht
#     0__[h]__0

#     1_______1

#     2_______2
    
#     3__[h]__3
#     """
#     def __init__(self,
#                  n_inputs: int,):
#         super(HeaterLayerMatrix_Side, self).__init__()
#         if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
#         self.phase_shift = nn.Parameter(2*torch.pi*torch.rand(n_inputs, device=device), requires_grad=True)
#         self.mask = torch.tensor([False for _ in range(n_inputs)], dtype=torch.bool)
#         self.mask[0] = True
#         self.mask[-1] = True
#         self.conn = torch.ones((n_inputs), device=device)

#     def forward(self, x_matrix):
#         layer = torch.where(self.mask, torch.exp(1.j*self.phase_shift), self.conn)
#         return layer.view(-1, 1) * x_matrix


# Forward clamping, backward return same gradiend without clamping
def clamp_ste(x, min=0.0, max=1.0):
    y = torch.clamp(x, min, max)     # forward value you want
    return x + (y - x).detach()    # forward = y, backward = identity


class PCLayerMatrix_Full(nn.Module):
    r"""
    Create waveguide matrix layer with all lines connected to a PhaseChanger with constant losses
        0__[PC]__0

        1__[PC]__1

        2__[PC]__2
        
        3__[PC]__3
    """
    def __init__(self,
                 n_inputs: int,
                 type_lim: str = "clamp_ste"):
        super(PCLayerMatrix_Full, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        self.type_lim = type_lim
        phase0_min = 1/4*np.pi
        phase0_max = 3/4*np.pi
        self.phase_shift = nn.Parameter(torch.rand(n_inputs, device=device) * (phase0_max-phase0_min)+phase0_min, requires_grad=True)

    def forward(self, x_matrix):
        if self.type_lim == "clamp_ste":
            phase_shift_0pi = clamp_ste(self.phase_shift, min=0., max=np.pi)
        elif self.type_lim == "clamp":
            phase_shift_0pi = torch.clamp(self.phase_shift, min=0., max=np.pi)
        elif self.type_lim == "tanh_soft":
            k = 0.5
            phase_shift_0pi = np.pi/2 * (torch.tanh(k * (self.phase_shift - np.pi/2) + 1))
        elif self.type_lim == "tanh_hard":
            k = 3.
            phase_shift_0pi = np.pi/2 * (torch.tanh(k * (self.phase_shift - np.pi/2) + 1))
        layer = torch.exp(1.j*phase_shift_0pi)
        return layer.view(-1, 1) * x_matrix



class PCLayerMatrix_Even(nn.Module):
    r"""
    Create waveguide matrix layer with just even lines connected to a PhaseChanger with constant losses
        0__[PC]__0

        1_______1

        2__[PC]__2
        
        3_______3
    """
    def __init__(self,
                 n_inputs: int,
                 type_lim: str = "clamp_ste"):
        super(PCLayerMatrix_Even, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        self.type_lim = type_lim
        phase0_min = 1/4*np.pi
        phase0_max = 3/4*np.pi
        self.phase_shift = nn.Parameter(torch.rand(n_inputs, device=device) * (phase0_max-phase0_min)+phase0_min, requires_grad=True)
        self.conn = torch.ones((n_inputs), device=device)
        self.mask = torch.tensor([i % 2 == 0 for i in range(n_inputs)], dtype=torch.bool)

    def forward(self, x_matrix):
        if self.type_lim == "clamp_ste":
            phase_shift_0pi = clamp_ste(self.phase_shift, min=0., max=np.pi)
        elif self.type_lim == "clamp":
            phase_shift_0pi = torch.clamp(self.phase_shift, min=0., max=np.pi)
        elif self.type_lim == "tanh_soft":
            k = 0.5
            phase_shift_0pi = np.pi/2 * (torch.tanh(k * (self.phase_shift - np.pi/2) + 1))
        elif self.type_lim == "tanh_hard":
            k = 3.
            phase_shift_0pi = np.pi/2 * (torch.tanh(k * (self.phase_shift - np.pi/2) + 1))
        layer = torch.where(self.mask, torch.exp(1.j*phase_shift_0pi), self.conn)
        return layer.view(-1, 1) * x_matrix



class PCLayerMatrix_Odd(nn.Module):
    r""" Create waveguide matrix layer with just odd lines connected to a PhaseChanger with constant losses
    0_______0

    1__[PC]__1

    2_______2
    
    3_______3
    """
    def __init__(self,
                 n_inputs: int,
                 type_lim: str = "clamp_ste"):
        super(PCLayerMatrix_Odd, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        self.type_lim = type_lim
        phase0_min = 1/4*np.pi
        phase0_max = 3/4*np.pi
        self.phase_shift = nn.Parameter(torch.rand(n_inputs, device=device) * (phase0_max-phase0_min)+phase0_min, requires_grad=True)
        self.conn = torch.ones((n_inputs), device=device)
        self.mask = torch.tensor([i % 2 == 1 for i in range(n_inputs)], dtype=torch.bool)
        self.mask[-1] = False

    def forward(self, x_matrix):
        if self.type_lim == "clamp_ste":
            phase_shift_0pi = clamp_ste(self.phase_shift, min=0., max=np.pi)
        elif self.type_lim == "clamp":
            phase_shift_0pi = torch.clamp(self.phase_shift, min=0., max=np.pi)
        elif self.type_lim == "tanh_soft":
            k = 0.5
            phase_shift_0pi = np.pi/2 * (torch.tanh(k * (self.phase_shift - np.pi/2) + 1))
        elif self.type_lim == "tanh_hard":
            k = 3.
            phase_shift_0pi = np.pi/2 * (torch.tanh(k * (self.phase_shift - np.pi/2) + 1))
        layer = torch.where(self.mask, torch.exp(1.j*phase_shift_0pi), self.conn)
        return layer.view(-1, 1) * x_matrix


class PCLayerMatrix_Side(nn.Module):
    r""" Create waveguide matrix layer with just the top and bottom lines connected to a PhaseChanger with constant losses
    0__[PC]__0

    1_______1

    2_______2
    
    3__[PC]__3
    """
    def __init__(self,
                 n_inputs: int,
                 type_lim: str = "clamp_ste"):
        super(PCLayerMatrix_Side, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        self.type_lim = type_lim
        phase0_min = 1/4*np.pi
        phase0_max = 3/4*np.pi
        self.phase_shift = nn.Parameter(torch.rand(n_inputs, device=device) * (phase0_max-phase0_min)+phase0_min, requires_grad=True)
        self.conn = torch.ones((n_inputs), device=device)
        self.mask = torch.tensor([False for _ in range(n_inputs)], dtype=torch.bool)
        self.mask[0] = True
        self.mask[-1] = True

    def forward(self, x_matrix):
        if self.type_lim == "clamp_ste":
            phase_shift_0pi = clamp_ste(self.phase_shift, min=0., max=np.pi)
        elif self.type_lim == "clamp":
            phase_shift_0pi = torch.clamp(self.phase_shift, min=0., max=np.pi)
        elif self.type_lim == "tanh_soft":
            k = 0.5
            phase_shift_0pi = np.pi/2 * (torch.tanh(k * (self.phase_shift - np.pi/2) + 1))
        elif self.type_lim == "tanh_hard":
            k = 3.
            phase_shift_0pi = np.pi/2 * (torch.tanh(k * (self.phase_shift - np.pi/2) + 1))
        layer = torch.where(self.mask, torch.exp(1.j*phase_shift_0pi), self.conn)
        return layer.view(-1, 1) * x_matrix
