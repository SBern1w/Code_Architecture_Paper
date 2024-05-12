import torch
import torch.nn as nn


# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class MMI(nn.Module):
    r"""
    Two port MMI with losses and imbalances
        0 __  __0                              | sqrt(1/2+IMB)   | j sqrt(1/2-IMB) |
            \/        =>   TF = sqrt(1-LOSS) x |                 |                 |
        1 __/\__1                              | j sqrt(1/2-IMB) |   sqrt(1/2+IMB) |

    TF = transfer function
    output_TFmatrix_2xN = TF_2x2 x input_TFmatrix_2xN

    n_MMI: number of MMIs
    i_loss: Tesor of size n_MMI, this is required for setting indepent losses to each MMI
    imbalance: Tensor of size n_bs, this is required for setting indepent imbalance to each beamsplitters.
    """

    def __init__(self,
                 n_mmi: int,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None):
        super(MMI, self).__init__()
        if mmi_i_losses is None:
            atten = torch.ones(n_mmi, device=device)
        else:
            atten = (1 - mmi_i_losses)
        if mmi_imbalances is None:
            balance1 = 0.5 * torch.ones(n_mmi, device=device)
            balance2 = 0.5 * torch.ones(n_mmi, device=device)
        else:
            balance1 = 0.5 + mmi_imbalances
            balance2 = 0.5 - mmi_imbalances
        self._mmi_tf_matrix = torch.zeros((n_mmi, 2, 2), dtype=torch.cfloat, device=device)
        self._mmi_tf_matrix[:, 0, 0] = atten * torch.sqrt(balance1)
        self._mmi_tf_matrix[:, 0, 1] = atten * 1.j * torch.sqrt(balance2)
        self._mmi_tf_matrix[:, 1, 0] = atten * 1.j * torch.sqrt(balance2)
        self._mmi_tf_matrix[:, 1, 1] = atten * torch.sqrt(balance1)

    def forward(self, input_matrix):
        # Batch([[a b], [c, d]] x [[A, B, C, D], [A, B, C, D]])
        return torch.bmm(self._mmi_tf_matrix, input_matrix)



class MMILayerMatrix_Even(nn.Module):
    r""" Create even MMIs matrix
        0 __  __0
            \/
        1 __/\__1
        
        2 __  __2
            \/
        3 __/\__3
    """
    def __init__(self,
                 n_inputs: int,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None):
        super(MMILayerMatrix_Even, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        self._n_inputs = n_inputs
        self._n_mmi = n_inputs//2
        self._mmis = MMI(
            n_mmi=self._n_mmi,
            mmi_i_losses=mmi_i_losses,
            mmi_imbalances=mmi_imbalances)
        
    def forward(self, x_matrix):
        y_matrix = torch.zeros_like(x_matrix, device=device)
        y_batch_matrix = self._mmis(x_matrix.view(self._n_mmi, 2, self._n_inputs))
        y_matrix = y_batch_matrix.view(2*self._n_mmi, -1)
        return y_matrix



class MMILayerMatrix_Odd(nn.Module):
    r""" Create odd MMIs matrix
    0 ______0

    1 __  __1
        \/
    2 __/\__2

    3 ______3
    """
    def __init__(self,
                 n_inputs: int,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None):
        super(MMILayerMatrix_Odd, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        self._n_inputs = n_inputs
        self._n_mmi = n_inputs//2 - 1
        self._mmis = MMI(
            n_mmi=self._n_mmi,
            mmi_i_losses=mmi_i_losses,
            mmi_imbalances=mmi_imbalances)
        
    def forward(self, x_matrix):
        y_matrix = torch.zeros_like(x_matrix, device=device)
        # Connect first and last conenctions
        y_matrix[0, :] = x_matrix[0, :]
        y_matrix[-1, :] = x_matrix[-1, :]
        # Derive the moltiplicpication reshape [n_mmi, 2, n_inputs]
        y_batch_matrix = self._mmis(x_matrix[1:-1, :].view(self._n_mmi, 2, self._n_inputs))
        # Reshape back to [2*n_mmi, n_inputs] and report to the outputs [1, n_inputs-1]
        y_matrix[1:-1, :] = y_batch_matrix.view(2*self._n_mmi, -1)
        return y_matrix


