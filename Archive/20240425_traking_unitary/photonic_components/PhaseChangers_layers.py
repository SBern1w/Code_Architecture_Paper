import torch
import torch.nn as nn

# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# class Connection(nn.Module):
#     r"""
#     Connection
#     0______0
    
#     output_wave = 1 x input_wave
#     """
#     def __init__(self):
#         super(Connection, self).__init__()

#     def forward(self, input_wave):       # input_wave = complex value
#         return input_wave



class Heater(nn.Module):
    r"""
    Heater
        0 __[h]__0
    
    output_wave = exp(j*phi) x input_wave
    """
    def __init__(self,
                 n_ht: int):
        super(Heater, self).__init__()
        phi = 2*torch.pi*torch.rand(n_ht, device=device)
        self.phase_shift = nn.Parameter(phi, requires_grad=True)

    def forward(self, input_matrix):       # input_wave = complex value
        # .view(-1, 1) make the vector [[a],[b],[c]] -> [a]*[input_row1], [b]*[input_row2], etc
        return torch.exp(1.j*self.phase_shift).view(-1, 1) * input_matrix



class HeaterLayerMatrix_Full(nn.Module):
    r"""
    Create waveguide matrix layer with all lines connected to a ht
        0 __[h]__0

        1 __[h]__1

        2 __[h]__2
        
        3 __[h]__3
    """
    def __init__(self,
                 n_inputs: int):
        super(HeaterLayerMatrix_Full, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        self._n_inputs = n_inputs
        self._heaters = Heater(n_ht=n_inputs)

    def forward(self, x_matrix):
        y_matrix = self._heaters(x_matrix)
        return y_matrix
    


class HeaterLayerMatrix_Even(nn.Module):
    r"""
    Create waveguide matrix layer with just even lines connected to a ht
        0 __[h]__0

        1 _______1

        2 __[h]__2
        
        3 _______3
    """
    def __init__(self,
                 n_inputs: int,):
        super(HeaterLayerMatrix_Even, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        self.n_inputs = n_inputs
        self._heaters = Heater(n_ht=n_inputs//2)

    def forward(self, x_matrix):
        y_matrix = torch.zeros_like(x_matrix, device=device)
        y_matrix[1::2] = x_matrix[1::2]
        y_matrix_heats = self._heaters(x_matrix[0::2])
        y_matrix[0::2] = y_matrix_heats
        return y_matrix


class HeaterLayerMatrix_Odd(nn.Module):
    r""" Create waveguide matrix layer with just odd lines connected to a ht
    0 _______0

    1 __[h]__1

    2 _______2
    
    3 _______3
    """
    def __init__(self,
                 n_inputs: int,):
        super(HeaterLayerMatrix_Odd, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        self.n_inputs = n_inputs
        self._heaters = Heater(n_ht=n_inputs//2-1)

    def forward(self, x_matrix):
        y_matrix = torch.zeros_like(x_matrix, device=device)
        y_matrix[0::2] = x_matrix[0::2]
        y_matrix[-1] = x_matrix[-1]
        y_matrix_heats = self._heaters(x_matrix[1:-1:2])
        y_matrix[1:-1:2] = y_matrix_heats
        return y_matrix


