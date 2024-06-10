import torch
import torch.nn as nn

# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class HeaterLayerMatrix_Full(nn.Module):
    r"""
    Create waveguide matrix layer with all lines connected to a ht
        0__[h]__0

        1__[h]__1

        2__[h]__2
        
        3__[h]__3
    """
    def __init__(self,
                 n_inputs: int):
        super(HeaterLayerMatrix_Full, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        self.phase_shift = nn.Parameter(2*torch.pi*torch.rand(n_inputs, device=device), requires_grad=True)

    def forward(self, x_matrix):
        layer = torch.exp(1.j*self.phase_shift)
        return layer.view(-1, 1) * x_matrix



class HeaterLayerMatrix_Even(nn.Module):
    r"""
    Create waveguide matrix layer with just even lines connected to a ht
        0__[h]__0

        1_______1

        2__[h]__2
        
        3_______3
    """
    def __init__(self,
                 n_inputs: int,):
        super(HeaterLayerMatrix_Even, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        self.phase_shift = nn.Parameter(2*torch.pi*torch.rand(n_inputs, device=device), requires_grad=True)
        self.conn = torch.ones((n_inputs), device=device)
        self.mask = torch.tensor([i % 2 == 0 for i in range(n_inputs)], dtype=torch.bool)

    def forward(self, x_matrix):
        layer = torch.where(self.mask, torch.exp(1.j*self.phase_shift), self.conn)
        return layer.view(-1, 1) * x_matrix



class HeaterLayerMatrix_Odd(nn.Module):
    r""" Create waveguide matrix layer with just odd lines connected to a ht
    0_______0

    1__[h]__1

    2_______2
    
    3_______3
    """
    def __init__(self,
                 n_inputs: int,):
        super(HeaterLayerMatrix_Odd, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        self.phase_shift = nn.Parameter(2*torch.pi*torch.rand(n_inputs, device=device), requires_grad=True)
        self.mask = torch.tensor([i % 2 == 1 for i in range(n_inputs)], dtype=torch.bool)
        self.mask[-1] = False
        self.conn = torch.ones((n_inputs), device=device)

    def forward(self, x_matrix):
        layer = torch.where(self.mask, torch.exp(1.j*self.phase_shift), self.conn)
        return layer.view(-1, 1) * x_matrix


class HeaterLayerMatrix_Side(nn.Module):
    r""" Create waveguide matrix layer with just the top and bottom lines connected to a ht
    0__[h]__0

    1_______1

    2_______2
    
    3__[h]__3
    """
    def __init__(self,
                 n_inputs: int,):
        super(HeaterLayerMatrix_Side, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        self.phase_shift = nn.Parameter(2*torch.pi*torch.rand(n_inputs, device=device), requires_grad=True)
        self.mask = torch.tensor([False for _ in range(n_inputs)], dtype=torch.bool)
        self.mask[0] = True
        self.mask[-1] = True
        self.conn = torch.ones((n_inputs), device=device)

    def forward(self, x_matrix):
        layer = torch.where(self.mask, torch.exp(1.j*self.phase_shift), self.conn)
        return layer.view(-1, 1) * x_matrix

