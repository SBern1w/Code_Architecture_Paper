import torch
import torch.nn as nn


# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class CrossingLayerMatrix_Odd(nn.Module):
    r"""
    Create odd Crossings matrix
        0__   __0
           \-/
        1__   __1
           \-/
        2__/-\__2
        
        3__/-\__3

    Where:
        0__   __0                              |   sqrt(CT)   | j sqrt(1-CT) |
           \-/        =>   TF = sqrt(1-LOSS) x |              |              |
        1__/-\__1                              | j sqrt(1-CT) |   sqrt(CT)   |
    """
    def __init__(self,
                 n_inputs: int,
                 crossing_i_losses: torch.Tensor = None,
                 crossing_crosstalks: torch.Tensor = None):
        super(CrossingLayerMatrix_Odd, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        _n_crossing = n_inputs//2 - 1
        if crossing_i_losses is None or crossing_crosstalks is None:
            atten = torch.ones(_n_crossing, device=device)
            tau = torch.zeros(_n_crossing, device=device)
            kappa = atten
            edge_up_crossing = torch.Tensor([1.])
            edge_dw_crossing = edge_up_crossing
        else:
            atten = (1 - crossing_i_losses[1:-1])
            tau = crossing_crosstalks[1:-1]
            kappa = 1 - crossing_crosstalks[1:-1]
            edge_up_crossing = (1 - crossing_i_losses[0])* torch.sqrt(torch.Tensor([crossing_crosstalks[0]]))
            edge_dw_crossing = (1 - crossing_i_losses[-1])* torch.sqrt(torch.Tensor([crossing_crosstalks[-1]]))

        tau_tf = atten * torch.sqrt(tau)
        kappa_tf = atten * 1.j * torch.sqrt(kappa)
        
        crossing_elements = torch.stack([
            torch.stack([tau_tf, kappa_tf], dim=-1),
            torch.stack([kappa_tf, tau_tf], dim=-1)
            ], dim=-2)

        self._crossing_tf_matrix = torch.block_diag(edge_up_crossing, *crossing_elements, edge_dw_crossing)
    
    def forward(self, x_matrix):
        return torch.mm(self._crossing_tf_matrix, x_matrix)