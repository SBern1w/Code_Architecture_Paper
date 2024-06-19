import torch
import torch.nn as nn


# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class MMILayerMatrix_Even(nn.Module):
    r"""
    Create even MMIs matrix
        0__  __0
           \/
        1__/\__1
        
        2__  __2
           \/
        3__/\__3

    Where:
        0__  __0                              | sqrt(1/2+IMB)   | j sqrt(1/2-IMB) |
           \/        =>   TF = sqrt(1-LOSS) x |                 |                 |
        1__/\__1                              | j sqrt(1/2-IMB) |   sqrt(1/2+IMB) |
    """
    def __init__(self,
                 n_inputs: int,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None):
        super(MMILayerMatrix_Even, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        _n_mmi = n_inputs//2
        if mmi_i_losses is None or mmi_imbalances is None:
            i_loss = torch.ones(_n_mmi, device=device)
            tau = 0.5 * i_loss
            kappa = tau
        else:
            i_loss = mmi_i_losses
            alpha = 1/2*(mmi_imbalances-1)/(mmi_imbalances+1)
            tau = 0.5 + alpha
            kappa = 0.5 - alpha

        tau_tf = torch.sqrt(i_loss) * torch.sqrt(tau)
        kappa_tf = torch.sqrt(i_loss) * 1.j * torch.sqrt(kappa)
        
        mmi_elements = torch.stack([
            torch.stack([tau_tf, kappa_tf], dim=-1),
            torch.stack([kappa_tf, tau_tf], dim=-1)
            ], dim=-2)
        
        self._mmi_tf_matrix = torch.block_diag(*mmi_elements)
    
    def forward(self, x_matrix):
        return torch.mm(self._mmi_tf_matrix, x_matrix)



class MMILayerMatrix_Odd(nn.Module):
    r"""
    Create odd MMIs matrix
        0______0

        1__  __1
           \/
        2__/\__2

        3______3

    Where:
        0__  __0                              | sqrt(1/2+IMB)   | j sqrt(1/2-IMB) |
           \/        =>   TF = sqrt(1-LOSS) x |                 |                 |
        1__/\__1                              | j sqrt(1/2-IMB) |   sqrt(1/2+IMB) |
    """
    def __init__(self,
                 n_inputs: int,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None):
        super(MMILayerMatrix_Odd, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        _n_mmi = n_inputs//2 - 1
        if mmi_i_losses is None or mmi_imbalances is None:
            i_loss = torch.ones(_n_mmi, device=device)
            tau = 0.5 * i_loss
            kappa = tau
        else:
            i_loss = mmi_i_losses
            alpha = 1/2*(mmi_imbalances-1)/(mmi_imbalances+1)
            tau = 0.5 + alpha
            kappa = 0.5 - alpha
        
        tau_tf = torch.sqrt(i_loss) * torch.sqrt(tau)
        kappa_tf = torch.sqrt(i_loss) * 1.j * torch.sqrt(kappa)
        
        mmi_elements = torch.stack([
            torch.stack([tau_tf, kappa_tf], dim=-1),
            torch.stack([kappa_tf, tau_tf], dim=-1)
            ], dim=-2)
        
        conn = torch.Tensor([1.])
        self._mmi_tf_matrix = torch.block_diag(conn, *mmi_elements, conn)

    def forward(self, x_matrix):
        return torch.mm(self._mmi_tf_matrix, x_matrix)



