import torch
import torch.nn as nn


# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class CrossingLayerMatrix_Odd(nn.Module):
    r"""
    Create odd Crossings matrix
        0_______0
        
        1__   __1
           \-/
        2__/-\__2
        
        3_______3

    Where:
        0__   __0                            |  sqrt(alpha)  | sqrt(1-alpha) |
           \-/        =>   TF = sqrt(LOSS) x |               |               |
        1__/-\__1                            | sqrt(1-alpha) |  sqrt(alpha)  |
    """
    def __init__(self,
                 n_inputs: int,
                 crossing_i_losses: torch.Tensor = None,
                 crossing_crosstalks: torch.Tensor = None):
        super(CrossingLayerMatrix_Odd, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        _n_crossing = n_inputs//2 - 1
        if crossing_i_losses is None or crossing_crosstalks is None:
            i_loss = torch.ones(_n_crossing, device=device)
            tau = torch.zeros(_n_crossing, device=device)
            kappa = i_loss
            edge_up_crossing = torch.Tensor([1.], device=device)
            edge_dw_crossing = edge_up_crossing
        else:
            i_loss = crossing_i_losses
            alpha = crossing_crosstalks/(1-crossing_crosstalks)
            tau = alpha
            kappa = 1 - alpha
            edge_up_crossing = torch.Tensor([1.], device=device)
            edge_dw_crossing = edge_up_crossing

        tau_tf = torch.sqrt(i_loss) * torch.sqrt(tau).to(torch.complex64)
        kappa_tf = torch.sqrt(i_loss) * torch.sqrt(kappa).to(torch.complex64)
        
        crossing_elements = torch.stack([
            torch.stack([tau_tf, kappa_tf], dim=-1),
            torch.stack([kappa_tf, tau_tf], dim=-1)
            ], dim=-2)

        self._crossing_tf_matrix = torch.block_diag(edge_up_crossing, *crossing_elements, edge_dw_crossing)
    
    def forward(self, x_matrix):
        return torch.mm(self._crossing_tf_matrix, x_matrix)



class CrossingLayerMatrix_Odd_CossingSide(nn.Module):
    r"""
    Create odd Crossings matrix with side crossing to simmetry
        0__   __0
           \-/
        1__   __1
           \-/
        2__/-\__2
        
        3__/-\__3

    Where:
        0__   __0                            |  sqrt(alpha)  | sqrt(1-alpha) |
           \-/        =>   TF = sqrt(LOSS) x |               |               |
        1__/-\__1                            | sqrt(1-alpha) |  sqrt(alpha)  |
    """
    def __init__(self,
                 n_inputs: int,
                 crossing_i_losses: torch.Tensor = None,
                 crossing_crosstalks: torch.Tensor = None):
        super(CrossingLayerMatrix_Odd_CossingSide, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        _n_crossing = n_inputs//2 - 1
        if crossing_i_losses is None or crossing_crosstalks is None:
            i_loss = torch.ones(_n_crossing, device=device)
            tau = torch.zeros(_n_crossing, device=device)
            kappa = i_loss
            edge_up_crossing = torch.Tensor([1.], device=device)
            edge_dw_crossing = edge_up_crossing
        else:
            i_loss = crossing_i_losses[1:-1]
            alpha = crossing_crosstalks[1:-1]/(1-crossing_crosstalks[1:-1])
            tau = alpha
            kappa = 1 - alpha
            edge_up_crossing = torch.sqrt(crossing_i_losses[0]) * torch.sqrt(1 - crossing_crosstalks[0]/(1-crossing_crosstalks[0]))
            edge_dw_crossing = torch.sqrt(crossing_i_losses[-1]) * torch.sqrt(1 - crossing_crosstalks[-1]/(1-crossing_crosstalks[-1]))

        tau_tf = torch.sqrt(i_loss) * torch.sqrt(tau).to(torch.complex64)
        kappa_tf = torch.sqrt(i_loss) * torch.sqrt(kappa).to(torch.complex64)
        
        crossing_elements = torch.stack([
            torch.stack([tau_tf, kappa_tf], dim=-1),
            torch.stack([kappa_tf, tau_tf], dim=-1)
            ], dim=-2)

        self._crossing_tf_matrix = torch.block_diag(edge_up_crossing, *crossing_elements, edge_dw_crossing)
    
    def forward(self, x_matrix):
        return torch.mm(self._crossing_tf_matrix, x_matrix)

