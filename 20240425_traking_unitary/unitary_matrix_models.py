import numpy as np
import torch
import torch.nn as nn

from photonic_components.PhaseChangers_layers import *
from photonic_components.MMI_layers import *

# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# =================================================================================================================
# ========================================== OPTICAL LAYERS for architectures =====================================
# =================================================================================================================

class _layer_matrix_ht_updown(nn.Module):
    r""" Create waveguide matrix layer with just first and last lines connected to a simple ht
    0__[]__0

    1______1

    2______2
    
    3__[]__3
    """
    def __init__(self, N: int = 2):
        super(_layer_matrix_ht_updown, self).__init__()
        self.phase_shift = nn.Parameter(2*torch.pi*torch.rand(N, device=device), requires_grad=True)
        # Waveguide connections
        self.wg_delete = torch.ones(N, requires_grad=False, device=device)  # Delete the waveguide from parameters
        self.add_conns = torch.zeros(N, requires_grad=False, device=device) # Add the connections
        for i in range(1, N-1, 1):
            self.wg_delete[i] = 0.0
            self.add_conns[i] = 1.0
        
    def forward(self):
        wg_array = torch.exp(1.j*self.phase_shift)
        wg_circuit_matrix = torch.diag(wg_array * self.wg_delete + self.add_conns)
        return wg_circuit_matrix


class _layerM_MMIs_even(nn.Module):
    r""" Create a MMI matrix with first pin connect to even imput
    0__  __0
       \/
    1__/\__1
    
    2__  __2
       \/
    3__/\__3
    """
    def __init__(self, N: int, insertion_loss_MMI: float, imbalance_MMI: float):
        super(_layerM_MMIs_even, self).__init__()
        if N%2 == 1: Exception('N is odd!!! NONONO, put it even!!!')
        self.N = N
        self._MMIs = nn.ModuleList([MMI(insertion_loss_MMI=insertion_loss_MMI,
                                        imbalance_MMI=imbalance_MMI)
                                        for _ in range(N//2)])
    
    def forward(self, input_TFmatrix):
        output_TFmatrix = torch.zeros_like(input_TFmatrix, dtype=torch.cfloat)
        for i, MMI in enumerate(self._MMIs):
            output_TFmatrix[2*i:2*i+2] = MMI(input_TFmatrix[2*i:2*i+2])
        return output_TFmatrix


class _layerM_MMIs_odd(nn.Module):
    r""" Create a MMI matrix with first pin connect to odd imput
    0______0

    1__  __1
       \/
    2__/\__2
    
    3______3
    """
    def __init__(self, N: int, insertion_loss_MMI: float, imbalance_MMI: float):
        super(_layerM_MMIs_odd, self).__init__()
        if N%2 == 1: Exception('N is odd!!! NONONO, put it even!!!')
        self.N = N
        self._MMIs_conns = nn.ModuleList()
        self._MMIs_conns.append(Connection())
        self._MMIs_conns.extend([MMI(insertion_loss_MMI=insertion_loss_MMI,
                                     imbalance_MMI=imbalance_MMI)
                                     for _ in range(N//2-1)])
        self._MMIs_conns.append(Connection())
    
    def forward(self, input_TFmatrix):
        output_TFmatrix = torch.zeros_like(input_TFmatrix)
        for i, MMI_conn in enumerate(self._MMIs_conns):
            if (i == 0) or (i == len(self._MMIs_conns)-1):
                output_TFmatrix[i] = MMI_conn(input_TFmatrix[i])
            else:
                output_TFmatrix[2*i:2*i+2] = MMI_conn(input_TFmatrix[2*i:2*i+2])
        return output_TFmatrix


class _layer_matrix_MMI_even_IN(nn.Module):
    r""" Create a MMI matrix with first pin connect to even imput for the INPUT
    0__  __0
       \/
       /\__1
    
    2__  __2
       \/
       /\__3
    """
    def __init__(self, N: int, insersion_loss_MMI: float, imbalance_MMI: float):
        super(_layer_matrix_MMI_even_IN, self).__init__()
        # 2x2 MMI transfer matrix with insersion loss and imbalance
        MMI_device = np.sqrt(1-insersion_loss_MMI)*torch.tensor([[np.sqrt(1/2+imbalance_MMI)],
                                                                 [1.j*np.sqrt(1/2-imbalance_MMI)]],
                                                                 requires_grad=False)
        # MMI connections
        self.MMI_matrix = torch.zeros(size=(2*N,N), dtype=torch.cfloat, requires_grad=False, device=device)
        for i in range(0, N, 1):
            self.MMI_matrix[2*i:2*i+2, i:i+1] = MMI_device
        if N%2 == 1:
            self.MMI_matrix[N-1, N-1] = 1.0
    
    def forward(self):
        return self.MMI_matrix


class _layer_matrix_MMI_even_OUT(nn.Module):
    r""" Create a MMI matrix with first pin connect to even imput for the OUTPUT
    0__  __0
       \/
    1__/\
    
    2__  __2
       \/
    3__/\
    """
    def __init__(self, N: int, insersion_loss_MMI: float, imbalance_MMI: float):
        super(_layer_matrix_MMI_even_OUT, self).__init__()
        # 2x2 MMI transfer matrix with insersion loss and imbalance
        MMI_device = np.sqrt(1-insersion_loss_MMI)*torch.tensor([[np.sqrt(1/2+imbalance_MMI), 1.j*np.sqrt(1/2-imbalance_MMI)]],
                                                                 requires_grad=False)
        # MMI connections
        self.MMI_matrix = torch.zeros(size=(N,2*N), dtype=torch.cfloat, requires_grad=False, device=device)
        for i in range(0, N, 1):
            self.MMI_matrix[i:i+1, 2*i:2*i+2] = MMI_device
        if N%2 == 1:
            self.MMI_matrix[N-1, N-1] = 1.0
    
    def forward(self):
        return self.MMI_matrix


class _layer_matrix_Crossing_odd(nn.Module):
    r""" Create a Crossing matrix with first pin connect to odd imput
    0__   __0
       \-/
    1__   __1
       \-/
    2__/-\__2
    
    3__/-\__3
    """
    def __init__(self, N: int, insersion_loss_Crossing: float, cross_talk_Crossing: float):
        super(_layer_matrix_Crossing_odd, self).__init__()
        # 2x2 Crossing transfer matrix with insersion loss and imbalance
        Crossing_device = np.sqrt(1-insersion_loss_Crossing)*torch.tensor([[np.sqrt(cross_talk_Crossing), 1.j*np.sqrt(1-cross_talk_Crossing)],
                                                                           [1.j*np.sqrt(1-cross_talk_Crossing), np.sqrt(cross_talk_Crossing)]],
                                                                           requires_grad=False)
        # Crossing connections
        self.Cross_matrix = torch.zeros(size=(N,N), dtype=torch.cfloat, requires_grad=False, device=device)
        self.Cross_matrix[0, 0] = np.sqrt(1-insersion_loss_Crossing)*np.sqrt(1-cross_talk_Crossing)             # Device on the side to simmetry
        for i in range(1, N-1, 2):
            self.Cross_matrix[i:i+2, i:i+2] = Crossing_device
        if N%2 == 0:
            self.Cross_matrix[-1, -1] = np.sqrt(1-insersion_loss_Crossing)*np.sqrt(1-cross_talk_Crossing)       # Device on the side to simmetry
    
    def forward(self):
        return self.Cross_matrix


# =================================================================================================================
# ========================================== OPTICAL ARCHITECTURES ================================================
# =================================================================================================================

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Tzamn read here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Architecture TODO
"""
    Clements architecture:
        0 __[]__  __[]__  __________________[]__  __[]__  __________________[]__0
                \/      \/                      \/      \/
        1 ______/\______/\__[]__  __[]__  ______/\______/\__[]__  __[]__  __[]__1
                                \/      \/                      \/      \/
        2 __[]__  __[]__  ______/\______/\__[]__  __[]__  ______/\______/\__[]__2
                \/      \/                      \/      \/
        3 ______/\______/\______________________/\______/\__________________[]__3

        The last layer it used to make just any unitary matrix. For the NN is not necessary!!

    Clements Bell architecture:
        0__[]__  __[]__  __________[]__________  __[]__  __________[]______[]__4
               \/      \/                      \/      \/
        1__[]__/\__[]__/\______  __[]__  ______/\__[]__/\______  __[]__  __[]__5
                               \/      \/                      \/      \/
        2__[]__  __[]__  ______/\__[]__/\______  __[]__  ______/\__[]__/\__[]__6
               \/      \/                      \/      \/
        3__[]__/\__[]__/\__________[]__________/\__[]__/\__________[]______[]__7
    
        Same here the first and last layer of heaters not necessary for the NN

    Fldzhyan architecture:
        0 __[]__  __________[]__  __________[]__  __________[]__  __________[]__0
                \/              \/              \/              \/
        1 ______/\__[]__  ______/\__[]__  ______/\__[]__  ______/\__[]__  __[]__1
                        \/              \/              \/              \/
        2 __[]__  ______/\__[]__  ______/\__[]__  ______/\__[]__  ______/\__[]__2
                \/              \/              \/              \/
        3 ______/\______________/\______________/\______________/\__________[]__3
    
        Same last ht layer

    Fldzhyan Bell architecture:
        0 __[]__  __[]__________  __[]__________  __[]__________  __[]______[]__0
                \/              \/              \/              \/
        1 __[]__/\__[]__  ______/\__[]__  ______/\__[]__  ______/\__[]__  __[]__1
                        \/              \/              \/              \/
        2 __[]__  __[]__/\______  __[]__/\______  __[]__/\______  __[]__/\__[]__2
                \/              \/              \/              \/
        3 __[]__/\__[]__________/\__[]__________/\__[]__________/\__[]______[]__3

        Same first last ht layer

"""


# Clements Architecture -------------------------------------------------------------------------------------------
class Clements_Arct(nn.Module):
    r"""
    Clements architecture:
        0 __[]__  __[]__  __________________[]__  __[]__  __________________[]__0
                \/      \/                      \/      \/
        1 ______/\______/\__[]__  __[]__  ______/\______/\__[]__  __[]__  __[]__1
                                \/      \/                      \/      \/
        2 __[]__  __[]__  ______/\______/\__[]__  __[]__  ______/\______/\__[]__2
                \/      \/                      \/      \/
        3 ______/\______/\______________________/\______/\__________________[]__3

        with:
            0__[]__1 = Phase Shifter

            3__  __2
               \/    =  MMI
            0__/\__1
    """
    def __init__(self,
                 n_inputs: int,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None):
        super(Clements_Arct, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        self._n_inputs = n_inputs
        _n_mmi_even = n_inputs//2
        _n_mmi_odd = n_inputs//2 - 1
        
        self._ht_layer_even = nn.ModuleList([HeaterLayerMatrix_Even(n_inputs=n_inputs) for _ in range(n_inputs)])
        # self._mmi_layer_even = nn.ModuleList()
        self._ht_layer_odd = nn.ModuleList([HeaterLayerMatrix_Odd(n_inputs=n_inputs) for _ in range(n_inputs)])
        # self._mmi_layer_odd = nn.ModuleList()
        
        # TO MODIFY !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # for i in range(n_inputs):
        #     if (mmi_i_losses is not None) and (mmi_imbalances is not None):
        #         idx_even = range(_n_mmi_even*i, _n_mmi_even*i+_n_mmi_even)
        #         self._mmi_layer_even.append(MMILayerMatrix_Even(
        #             n_inputs=n_inputs,
        #             mmi_i_losses=mmi_i_losses[idx_even],
        #             mi_imbalances=mmi_imbalances[idx_even]))
                
        #         idx_odd = range(_n_mmi_odd*i+n_inputs*_n_mmi_even, _n_mmi_odd*i+n_inputs*_n_mmi_even+_n_mmi_odd)
        #         self._mmi_layer_odd.append(MMILayerMatrix_Odd(
        #             n_inputs=n_inputs,
        #             mmi_i_losses=mmi_i_losses[idx_odd],
        #             mi_imbalances=mmi_imbalances[idx_odd]))
        
        self._mmi_layer_even = nn.ModuleList([MMILayerMatrix_Even(n_inputs=n_inputs) for _ in range(n_inputs)])
        self._mmi_layer_odd = nn.ModuleList([MMILayerMatrix_Odd(n_inputs=n_inputs) for _ in range(n_inputs)])
                # self._mmi_layer_odd.append(MMILayerMatrix_Odd(n_inputs=n_inputs))
        
        self._ht_layer_out = HeaterLayerMatrix_Full(n_inputs=n_inputs)

    def forward(self):
        id_ini = torch.eye(self._n_inputs, device=device)      # Identity matrix
        arct_matrix = torch.complex(id_ini, torch.zeros_like(id_ini, device=device))
        for i in range(self._n_inputs//2):
            arct_matrix = self._ht_layer_even[2*i](arct_matrix)
            arct_matrix = self._mmi_layer_even[2*i](arct_matrix)
            arct_matrix = self._ht_layer_even[2*i+1](arct_matrix)
            arct_matrix = self._mmi_layer_even[2*i+1](arct_matrix)

            arct_matrix = self._ht_layer_odd[2*i](arct_matrix)
            arct_matrix = self._mmi_layer_odd[2*i](arct_matrix)
            arct_matrix = self._ht_layer_odd[2*i+1](arct_matrix)
            arct_matrix = self._mmi_layer_odd[2*i+1](arct_matrix)
        
        arct_matrix = self._ht_layer_out(arct_matrix)
        return arct_matrix



# Clements Bell Architecture --------------------------------------------------------------------------------------
class ClementsBell_Arct(nn.Module):
    r"""
    Clements Bell architecture:
        0__[]__  __[]__  ______________________  __[]__  __________________[]__4
               \/      \/                      \/      \/
        1__[]__/\__[]__/\______  __[]__  ______/\__[]__/\______  __[]__  __[]__5
                               \/      \/                      \/      \/
        2__[]__  __[]__  ______/\__[]__/\______  __[]__  ______/\__[]__/\__[]__6
               \/      \/                      \/      \/
        3__[]__/\__[]__/\______________________/\__[]__/\__________________[]__7

        with:
            0__[]__1 = phase shift

            3__  __2
               \/    =  MMI
            0__/\__1
    """
    def __init__(self,
                 n_inputs: int,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None):
        super(ClementsBell_Arct, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        self._n_inputs = n_inputs
        _n_mmi_even = n_inputs//2
        _n_mmi_odd = n_inputs//2 - 1

        self._layer_ht = nn.ModuleList([HeaterLayerMatrix_Full(n_inputs=N) for i in range(N+2)])
        self._layer_MMI_even = nn.ModuleList([MMILayerMatrix_Even(n_inputs=N) for _ in range(N)])
        self._layer_MMI_odd = nn.ModuleList([MMILayerMatrix_Odd(n_inputs=N) for _ in range(N)])

    def forward(self):
        id_ini = torch.eye(self.N, requires_grad=False)      # Identity matrix
        arch_matrix = torch.complex(id_ini, torch.zeros_like(id_ini))
        arch_matrix = self._layer_ht[0](arch_matrix)
        for i in range(self.N//2):
            arch_matrix = self._layer_MMI_even[2*i](arch_matrix)
            arch_matrix = self._layer_ht[2*i+1](arch_matrix)
            arch_matrix = self._layer_MMI_even[2*i+1](arch_matrix)
            arch_matrix = self._layer_MMI_odd[2*i](arch_matrix)
            arch_matrix = self._layer_ht[2*i+2](arch_matrix)
            arch_matrix = self._layer_MMI_odd[2*i+1](arch_matrix)
        arch_matrix = self._layer_ht[2*i+3](arch_matrix)
        return arch_matrix


# FldzhyanBell ====================================================================================================
class FldzhyanBellNxN(nn.Module):
    r""" Fldzhyan Bell architecture
    Network:
        <----------------------------------2N----------------------------------> Length number of layer MMIs
        0__[]__  __[]__________  __[]__________  __[]__________  __[]______[]__0
               \/              \/              \/              \/
        1__[]__/\__[]__  ______/\__[]__  ______/\__[]__  ______/\__[]__  __[]__1
                       \/              \/              \/              \/
        2__[]__  __[]__/\______  __[]__/\______  __[]__/\______  __[]__/\__[]__2
               \/              \/              \/              \/
        3__[]__/\__[]__________/\__[]__________/\__[]__________/\__[]______[]__3

        with:
            0__[]__1 = phase shift

            3__  __2
               \/    =  MMI
            0__/\__1
    """
    def __init__(self, N: int, insersion_loss_MMI: float, imbalance_MMI: float):
        super(FldzhyanBellNxN, self).__init__()
        self.N = N
        if N%2 == 0:
            self._layer_ht = nn.ModuleList([_layer_matrix_ht_full(N=N) for i in range(N+2)])
            self._layer_MMI_even = nn.ModuleList([_layer_matrix_MMI_even(N=N,
                                                                         insersion_loss_MMI=insersion_loss_MMI,
                                                                         imbalance_MMI=imbalance_MMI) for i in range(N)])
            self._layer_MMI_odd = nn.ModuleList([_layer_matrix_MMI_odd(N=N,
                                                                       insersion_loss_MMI=insersion_loss_MMI,
                                                                       imbalance_MMI=imbalance_MMI) for i in range(N)])
        else:
            raise Exception('N is odd put it even')       # I don't want to thinking about not relevant

    def forward(self):
        id_ini = torch.eye(self.N, requires_grad=False)      # Identity matrix
        arch_matrix = torch.complex(id_ini, torch.zeros_like(id_ini))
        arch_matrix = torch.mm(self._layer_ht[0](), arch_matrix)
        for i in range(self.N):
            arch_matrix = torch.mm(self._layer_MMI_even[i](), arch_matrix)
            arch_matrix = torch.mm(self._layer_ht[i+1](), arch_matrix)
            arch_matrix = torch.mm(self._layer_MMI_odd[i](), arch_matrix)
        arch_matrix = torch.mm(self._layer_ht[i+2](), arch_matrix)
        return arch_matrix


# FldzhyanBellHalf ================================================================================================
class FldzhyanBellHalfNxN(nn.Module):
    r""" Fldzhyan Bell Half architecture
    Network:
        <-------------------N------------------> Length number of layer MMIs
        0__[]__  __[]__________  __[]______[]__0
               \/              \/
        1__[]__/\__[]__  ______/\__[]__  __[]__1
                       \/              \/
        2__[]__  __[]__/\______  __[]__/\__[]__2
               \/              \/
        3__[]__/\__[]__________/\__[]______[]__3

        with:
            0__[]__1 = phase shift

            3__  __2
               \/    =  MMI
            0__/\__1
    """
    def __init__(self, N: int, insersion_loss_MMI: float, imbalance_MMI: float):
        super(FldzhyanBellHalfNxN, self).__init__()
        self.N = N
        if N%2 == 0:
            self._layer_ht = nn.ModuleList([_layer_matrix_ht_full(N=N) for i in range(N//2+2)])
            self._layer_MMI_even = nn.ModuleList([_layer_matrix_MMI_even(N=N,
                                                                         insersion_loss_MMI=insersion_loss_MMI,
                                                                         imbalance_MMI=imbalance_MMI) for i in range(N//2)])
            self._layer_MMI_odd = nn.ModuleList([_layer_matrix_MMI_odd(N=N,
                                                                       insersion_loss_MMI=insersion_loss_MMI,
                                                                       imbalance_MMI=imbalance_MMI) for i in range(N//2)])
        else:
            raise Exception('N is odd put it even')       # I don't want to thinking about not relevant

    def forward(self):
        id_ini = torch.eye(self.N, requires_grad=False)      # Identity matrix
        arch_matrix = torch.complex(id_ini, torch.zeros_like(id_ini))
        arch_matrix = torch.mm(self._layer_ht[0](), arch_matrix)
        for i in range(self.N//2):
            arch_matrix = torch.mm(self._layer_MMI_even[i](), arch_matrix)
            arch_matrix = torch.mm(self._layer_ht[i+1](), arch_matrix)
            arch_matrix = torch.mm(self._layer_MMI_odd[i](), arch_matrix)
        arch_matrix = torch.mm(self._layer_ht[i+2](), arch_matrix)
        return arch_matrix


# NEUROPULSBonus_unitary ==========================================================================================
class NEUROPULSBonus_unitaryNxN(nn.Module):
    r""" NEUROPULS Bonus unitary architecture
    Network:
        <-----------------------N-----------------------> Length number of layer MMIs
        0__[]__  __[]__  ______   ______  __[]__  __[]__0
               \/      \/      \-/      \/      \/
        1__[]__/\__[]__/\______   ______/\__[]__/\__[]__1
                               \-/
        2__[]__  __[]__  ______/-\______  __[]__  __[]__2
               \/      \/               \/      \/
        3__[]__/\__[]__/\______/-\______/\__[]__/\__[]__3

        with:
            0__[]__1 = phase shift

            3__  __2
               \/    =  MMI
            0__/\__1

            3__   __2
               \-/   =  Crossing
            0__/-\__1
    """
    def __init__(self, N: int, insersion_loss_MMI: float, imbalance_MMI: float, insersion_loss_Crossing:float, cross_talk_Crossing: float):
        super(NEUROPULSBonus_unitaryNxN, self).__init__()
        self.N = N
        if N%2 == 0:
            self._layer_ht = nn.ModuleList([_layer_matrix_ht_full(N=N) for i in range(N//2+2)])
            self._layer_MMI_even = nn.ModuleList([_layer_matrix_MMI_even(N=N,
                                                                         insersion_loss_MMI=insersion_loss_MMI,
                                                                         imbalance_MMI=imbalance_MMI) for i in range(N//2)])
            
            self._layer_Crossing_odd = nn.ModuleList([_layer_matrix_Crossing_odd(N=N,
                                                                                 insersion_loss_Crossing=insersion_loss_Crossing,
                                                                                 cross_talk_Crossing=cross_talk_Crossing) for i in range(N//2-1)])
        else:
            raise Exception('N is odd put it even')       # I don't want to thinking about not relevant

    def forward(self):
        id_ini = torch.eye(self.N, requires_grad=False)      # Identity matrix
        arch_matrix = torch.complex(id_ini, torch.zeros_like(id_ini))
        arch_matrix = torch.mm(self._layer_ht[0](), arch_matrix)
        for i in range(self.N//2):
            arch_matrix = torch.mm(self._layer_MMI_even[i](), arch_matrix)
            arch_matrix = torch.mm(self._layer_ht[i+1](), arch_matrix)
            arch_matrix = torch.mm(self._layer_MMI_even[i](), arch_matrix)
            if i < self.N//2-1:
                arch_matrix = torch.mm(self._layer_Crossing_odd[i](), arch_matrix)
        arch_matrix = torch.mm(self._layer_ht[i+2](), arch_matrix)
        return arch_matrix


# NEUROPULSBonus_2long_unitary ====================================================================================
class NEUROPULSBonus_unitary_2long_NxN(nn.Module):
    r""" NEUROPULS Bonus unitary architecture
    Network:
        <-------------------------------------------------2N----------------------------------------------> Length number of layer MMIs
        0__[]__  __[]__  ______   ______  __[]__  ______   ______  __[]__  ______   ______  __[]__  __[]__0
               \/      \/      \-/      \/      \/      \-/      \/      \/      \-/      \/      \/
        1__[]__/\__[]__/\______   ______/\__[]__/\______   ______/\__[]__/\______   ______/\__[]__/\__[]__1
                               \-/                      \-/                      \-/
        2__[]__  __[]__  ______/-\______  __[]__  ______/-\______  __[]__  ______/-\______  __[]__  __[]__2
               \/      \/               \/      \/               \/      \/               \/      \/
        3__[]__/\__[]__/\______/-\______/\__[]__/\______/-\______/\__[]__/\______/-\______/\__[]__/\__[]__3

        with:
            0__[]__1 = phase shift

            3__  __2
               \/    =  MMI
            0__/\__1

            3__   __2
               \-/   =  Crossing
            0__/-\__1
    """
    def __init__(self, N: int, insersion_loss_MMI: float, imbalance_MMI: float, insersion_loss_Crossing:float, cross_talk_Crossing: float):
        super(NEUROPULSBonus_unitary_2long_NxN, self).__init__()
        self.N = N
        if N%2 == 0:
            self._layer_ht = nn.ModuleList([_layer_matrix_ht_full(N=N) for i in range(N+2)])
            self._layer_MMI_even = nn.ModuleList([_layer_matrix_MMI_even(N=N,
                                                                         insersion_loss_MMI=insersion_loss_MMI,
                                                                         imbalance_MMI=imbalance_MMI) for i in range(2*N)])
            
            self._layer_Crossing_odd = nn.ModuleList([_layer_matrix_Crossing_odd(N=N,
                                                                                 insersion_loss_Crossing=insersion_loss_Crossing,
                                                                                 cross_talk_Crossing=cross_talk_Crossing) for i in range(N-1)])
        else:
            raise Exception('N is odd put it even')       # I don't want to thinking about not relevant

    def forward(self):
        id_ini = torch.eye(self.N, requires_grad=False)      # Identity matrix
        arch_matrix = torch.complex(id_ini, torch.zeros_like(id_ini))
        arch_matrix = torch.mm(self._layer_ht[0](), arch_matrix)
        for i in range(self.N):
            arch_matrix = torch.mm(self._layer_MMI_even[2*i](), arch_matrix)
            arch_matrix = torch.mm(self._layer_ht[i+1](), arch_matrix)
            arch_matrix = torch.mm(self._layer_MMI_even[2*i+1](), arch_matrix)
            if i < self.N-1:
                arch_matrix = torch.mm(self._layer_Crossing_odd[i](), arch_matrix)
        arch_matrix = torch.mm(self._layer_ht[i+2](), arch_matrix)
        return arch_matrix


# NEUROPULSBonus_anymatrix ========================================================================================
# Seems it work fine just if the sum|row/column|^2<0.5
# Make sense because the last layer we lose half of the power
class NEUROPULSBonus_anymatrixNxN(nn.Module):
    r""" NEUROPULS Bonus any matrix architecture
    Network:
        <-----------------------------------------------2N-------------------------------------------------> Length number of layer MMIs
        0__[]__  __[]__  ______   __[]__  __[]__  ______   __[]__  __[]__  ______   __[]__  __[]__  __[]__0
               \/      \/      \-/      \/      \/      \-/      \/      \/      \-/      \/      \/
             __/\______/\______   ______/\______/\______   ______/\______/\______   ______/\______/\
                               \-/                      \-/                      \-/
        1__[]__  __[]__  ______/-\__[]__  __[]__  ______/-\__[]__  __[]__  ______/-\__[]__  __[]__  __[]__1
               \/      \/               \/      \/               \/      \/               \/      \/
             __/\______/\______   ______/\______/\______   ______/\______/\______   ______/\______/\
                               \-/                      \-/                      \-/
        2__[]__  __[]__  ______/-\__[]__  __[]__  ______/-\__[]__  __[]__  ______/-\__[]__  __[]__  __[]__2
               \/      \/               \/      \/               \/      \/               \/      \/
             __/\______/\______   ______/\______/\______   ______/\______/\______   ______/\______/\
                               \-/                      \-/                      \-/
        3__[]__  __[]__  ______/-\__[]__  __[]__  ______/-\__[]__  __[]__  ______/-\__[]__  __[]__  __[]__3
               \/      \/               \/      \/               \/      \/               \/      \/
             __/\______/\______/-\______/\______/\______/-\______/\______/\______/-\______/\______/\

        with:
            0__[]__1 = phase shift

            3__  __2
               \/    =  MMI
            0__/\__1

            3__   __2
               \-/   =  Crossing
            0__/-\__1
    """
    def __init__(self, N: int, insersion_loss_MMI: float, imbalance_MMI: float, insersion_loss_Crossing:float, cross_talk_Crossing: float):
        super(NEUROPULSBonus_anymatrixNxN, self).__init__()
        self.N = N
        if N%2 == 0:
            self._layer_ht_IN = _layer_matrix_ht_full(N=N)
            self._layer_MMI_IN = _layer_matrix_MMI_even_IN(N=N,
                                                           insersion_loss_MMI=insersion_loss_MMI,
                                                           imbalance_MMI=imbalance_MMI)
            self._layer_ht_even = nn.ModuleList([_layer_matrix_ht_even(N=2*N) for i in range(2*N-1)])
            self._layer_MMI_even = nn.ModuleList([_layer_matrix_MMI_even(N=2*N,
                                                                         insersion_loss_MMI=insersion_loss_MMI,
                                                                         imbalance_MMI=imbalance_MMI) for i in range(2*N-2)])
            self._layer_Crossing_odd = nn.ModuleList([_layer_matrix_Crossing_odd(N=2*N,
                                                                                 insersion_loss_Crossing=insersion_loss_Crossing,
                                                                                 cross_talk_Crossing=cross_talk_Crossing) for i in range(N-1)])
            
            self._layer_MMI_OUT = _layer_matrix_MMI_even_OUT(N=N,
                                                             insersion_loss_MMI=insersion_loss_MMI,
                                                             imbalance_MMI=imbalance_MMI)
            self._layer_ht_OUT = _layer_matrix_ht_full(N=N)
        else:
            raise Exception('N is odd put it even')       # I don't want to thinking about not relevant

    def forward(self):
        id_ini = torch.eye(self.N, requires_grad=False)      # Identity matrix
        arch_matrix = torch.complex(id_ini, torch.zeros_like(id_ini))
        for i in range(self.N):
            if i == 0:          # INPUT
                arch_matrix = torch.mm(self._layer_ht_IN(), arch_matrix)
                arch_matrix = torch.mm(self._layer_MMI_IN(), arch_matrix)
                arch_matrix = torch.mm(self._layer_ht_even[0](), arch_matrix)
                arch_matrix = torch.mm(self._layer_MMI_even[0](), arch_matrix)
                arch_matrix = torch.mm(self._layer_Crossing_odd[0](), arch_matrix)
            elif i < self.N-1:  # CENTER
                arch_matrix = torch.mm(self._layer_ht_even[2*i-1](), arch_matrix)
                arch_matrix = torch.mm(self._layer_MMI_even[2*i-1](), arch_matrix)
                arch_matrix = torch.mm(self._layer_ht_even[2*i](), arch_matrix)
                arch_matrix = torch.mm(self._layer_MMI_even[2*i](), arch_matrix)
                arch_matrix = torch.mm(self._layer_Crossing_odd[i](), arch_matrix)
            else:               # OUTPUT
                arch_matrix = torch.mm(self._layer_ht_even[2*i-1](), arch_matrix)
                arch_matrix = torch.mm(self._layer_MMI_even[2*i-1](), arch_matrix)
                arch_matrix = torch.mm(self._layer_ht_even[2*i](), arch_matrix)
                arch_matrix = torch.mm(self._layer_MMI_OUT(), arch_matrix)
                arch_matrix = torch.mm(self._layer_ht_OUT(), arch_matrix)
        return arch_matrix



# NEUROPULSBonus_anymatrix ========================================================================================
class NEUROPULSBonus_Bell_Minht_NxN(nn.Module):
    r""" NEUROPULS Bonus any matrix architecture
    Network:
        <-----------------------------------------------2N-------------------------------------------------> Length number of layer MMIs
        0__[]__  __[]__  ______   __[]__  __[]__  ______   __[]__  __[]__  ______   __[]__  __[]__  __[]__0
               \/      \/      \-/      \/      \/      \-/      \/      \/      \-/      \/      \/
             __/\__[]__/\______   ______/\__[]__/\______   ______/\__[]__/\______   ______/\__[]__/\
                               \-/                      \-/                      \-/
        1__[]__  __[]__  ______/-\______  __[]__  ______/-\______  __[]__  ______/-\______  __[]__  __[]__1
               \/      \/               \/      \/               \/      \/               \/      \/
             __/\__[]__/\______   ______/\__[]__/\______   ______/\__[]__/\______   ______/\__[]__/\
                               \-/                      \-/                      \-/
        2__[]__  __[]__  ______/-\______  __[]__  ______/-\______  __[]__  ______/-\______  __[]__  __[]__2
               \/      \/               \/      \/               \/      \/               \/      \/
             __/\__[]__/\______   ______/\__[]__/\______   ______/\__[]__/\______   ______/\__[]__/\
                               \-/                      \-/                      \-/
        3__[]__  __[]__  ______/-\______  __[]__  ______/-\______  __[]__  ______/-\______  __[]__  __[]__3
               \/      \/               \/      \/               \/      \/               \/      \/
             __/\__[]__/\______/-\__[]__/\__[]__/\______/-\__[]__/\__[]__/\______/-\__[]__/\__[]__/\

        with:
            0__[]__1 = phase shift

            3__  __2
               \/    =  MMI
            0__/\__1

            3__   __2
               \-/   =  Crossing
            0__/-\__1
    """
    def __init__(self, N: int, insersion_loss_MMI: float, imbalance_MMI: float, insersion_loss_Crossing:float, cross_talk_Crossing: float):
        super(NEUROPULSBonus_Bell_Minht_NxN, self).__init__()
        self.N = N
        if N%2 == 0:
            self._layer_ht_IN = _layer_matrix_ht_full(N=N)
            self._layer_MMI_IN = _layer_matrix_MMI_even_IN(N=N,
                                                           insersion_loss_MMI=insersion_loss_MMI,
                                                           imbalance_MMI=imbalance_MMI)
            self._layer_ht_full = nn.ModuleList([_layer_matrix_ht_full(N=2*N) for i in range(N)])
            self._layer_ht_updwn = nn.ModuleList([_layer_matrix_ht_updown(N=2*N) for i in range(N-1)])
            self._layer_MMI_even = nn.ModuleList([_layer_matrix_MMI_even(N=2*N,
                                                                         insersion_loss_MMI=insersion_loss_MMI,
                                                                         imbalance_MMI=imbalance_MMI) for i in range(2*N-2)])
            self._layer_Crossing_odd = nn.ModuleList([_layer_matrix_Crossing_odd(N=2*N,
                                                                                 insersion_loss_Crossing=insersion_loss_Crossing,
                                                                                 cross_talk_Crossing=cross_talk_Crossing) for i in range(N-1)])
            
            self._layer_MMI_OUT = _layer_matrix_MMI_even_OUT(N=N,
                                                             insersion_loss_MMI=insersion_loss_MMI,
                                                             imbalance_MMI=imbalance_MMI)
            self._layer_ht_OUT = _layer_matrix_ht_full(N=N)
        else:
            raise Exception('N is odd put it even')       # I don't want to thinking about not relevant

    def forward(self):
        id_ini = torch.eye(self.N, requires_grad=False)      # Identity matrix
        arch_matrix = torch.complex(id_ini, torch.zeros_like(id_ini))
        for i in range(self.N):
            if i == 0:          # INPUT
                arch_matrix = torch.mm(self._layer_MMI_IN(), arch_matrix)
                arch_matrix = torch.mm(self._layer_ht_full[0](), arch_matrix)
                arch_matrix = torch.mm(self._layer_MMI_even[0](), arch_matrix)
                arch_matrix = torch.mm(self._layer_Crossing_odd[0](), arch_matrix)
            elif i < self.N-1:  # CENTER
                arch_matrix = torch.mm(self._layer_ht_updwn[i-1](), arch_matrix)
                arch_matrix = torch.mm(self._layer_MMI_even[2*i-1](), arch_matrix)
                arch_matrix = torch.mm(self._layer_ht_full[i](), arch_matrix)
                arch_matrix = torch.mm(self._layer_MMI_even[2*i](), arch_matrix)
                arch_matrix = torch.mm(self._layer_Crossing_odd[i](), arch_matrix)
            else:               # OUTPUT
                arch_matrix = torch.mm(self._layer_ht_updwn[i-1](), arch_matrix)
                arch_matrix = torch.mm(self._layer_MMI_even[2*i-1](), arch_matrix)
                arch_matrix = torch.mm(self._layer_ht_full[i](), arch_matrix)
                arch_matrix = torch.mm(self._layer_MMI_OUT(), arch_matrix)
                arch_matrix = torch.mm(self._layer_ht_OUT(), arch_matrix)
        return arch_matrix



# NEUROPULSBonus_BellNormal =======================================================================================
class NEUROPULSBonus_BellNormal_NxN(nn.Module):
    r""" NEUROPULS Bonus any matrix architecture
    Network:
        <-----------------------------------------------2N-------------------------------------------------> Length number of layer MMIs
        0__[]__  __[]__  ______   ______  __[]__  ______   ______  __[]__  ______   ______  __[]__  __[]__0
               \/      \/      \-/      \/      \/      \-/      \/      \/      \-/      \/      \/
             __/\__[]__/\______   ______/\__[]__/\______   ______/\__[]__/\______   ______/\__[]__/\
                               \-/                      \-/                      \-/
        1__[]__  __[]__  ______/-\______  __[]__  ______/-\______  __[]__  ______/-\______  __[]__  __[]__1
               \/      \/               \/      \/               \/      \/               \/      \/
             __/\__[]__/\______   ______/\__[]__/\______   ______/\__[]__/\______   ______/\__[]__/\
                               \-/                      \-/                      \-/
        2__[]__  __[]__  ______/-\______  __[]__  ______/-\______  __[]__  ______/-\______  __[]__  __[]__2
               \/      \/               \/      \/               \/      \/               \/      \/
             __/\__[]__/\______   ______/\__[]__/\______   ______/\__[]__/\______   ______/\__[]__/\
                               \-/                      \-/                      \-/
        3__[]__  __[]__  ______/-\______  __[]__  ______/-\______  __[]__  ______/-\______  __[]__  __[]__3
               \/      \/               \/      \/               \/      \/               \/      \/
             __/\__[]__/\______/-\______/\__[]__/\______/-\______/\__[]__/\______/-\______/\__[]__/\

        with:
            0__[]__1 = phase shift

            3__  __2
               \/    =  MMI
            0__/\__1

            3__   __2
               \-/   =  Crossing
            0__/-\__1
    """
    def __init__(self, N: int, insersion_loss_MMI: float, imbalance_MMI: float, insersion_loss_Crossing:float, cross_talk_Crossing: float):
        super(NEUROPULSBonus_BellNormal_NxN, self).__init__()
        self.N = N
        if N%2 == 0:
            self._layer_ht_IN = _layer_matrix_ht_full(N=N)
            self._layer_MMI_IN = _layer_matrix_MMI_even_IN(N=N,
                                                           insersion_loss_MMI=insersion_loss_MMI,
                                                           imbalance_MMI=imbalance_MMI)
            self._layer_ht_full = nn.ModuleList([_layer_matrix_ht_full(N=2*N) for i in range(N)])
            self._layer_MMI_even = nn.ModuleList([_layer_matrix_MMI_even(N=2*N,
                                                                         insersion_loss_MMI=insersion_loss_MMI,
                                                                         imbalance_MMI=imbalance_MMI) for i in range(2*N-2)])
            self._layer_Crossing_odd = nn.ModuleList([_layer_matrix_Crossing_odd(N=2*N,
                                                                                 insersion_loss_Crossing=insersion_loss_Crossing,
                                                                                 cross_talk_Crossing=cross_talk_Crossing) for i in range(N-1)])
            self._layer_MMI_OUT = _layer_matrix_MMI_even_OUT(N=N,
                                                             insersion_loss_MMI=insersion_loss_MMI,
                                                             imbalance_MMI=imbalance_MMI)
            self._layer_ht_OUT = _layer_matrix_ht_full(N=N)
        else:
            raise Exception('N is odd put it even')       # I don't want to thinking about not relevant

    def forward(self):
        id_ini = torch.eye(self.N, requires_grad=False)      # Identity matrix
        arch_matrix = torch.complex(id_ini, torch.zeros_like(id_ini))
        for i in range(self.N):
            if i == 0:          # INPUT
                arch_matrix = torch.mm(self._layer_ht_IN(), arch_matrix)
                arch_matrix = torch.mm(self._layer_MMI_IN(), arch_matrix)
                arch_matrix = torch.mm(self._layer_ht_full[0](), arch_matrix)
                arch_matrix = torch.mm(self._layer_MMI_even[0](), arch_matrix)
                arch_matrix = torch.mm(self._layer_Crossing_odd[0](), arch_matrix)
            elif i < self.N-1:  # CENTER
                arch_matrix = torch.mm(self._layer_MMI_even[2*i-1](), arch_matrix)
                arch_matrix = torch.mm(self._layer_ht_full[i](), arch_matrix)
                arch_matrix = torch.mm(self._layer_MMI_even[2*i](), arch_matrix)
                arch_matrix = torch.mm(self._layer_Crossing_odd[i](), arch_matrix)
            else:               # OUTPUT
                arch_matrix = torch.mm(self._layer_MMI_even[2*i-1](), arch_matrix)
                arch_matrix = torch.mm(self._layer_ht_full[i](), arch_matrix)
                arch_matrix = torch.mm(self._layer_MMI_OUT(), arch_matrix)
                arch_matrix = torch.mm(self._layer_ht_OUT(), arch_matrix)
        return arch_matrix




# =================================================================================================================
# =========================== OPTICAL ARCHITECTURES Last layer a MZM ==============================================
# =================================================================================================================

# Arch_lastD_MZI =======================================================================================
class Arch_lastMZI_NxN(nn.Module):
    r""" Architecture
    Network:
            -----------
        0__|           |__[MZM]__4
           |           |
        1__|           |__[MZM]__5
           | Archeture |
        2__|           |__[MZM]__6
           |           |
        3__|           |__[MZM]__7
            -----------

        with:
            0__  __[]__  __1
               \/      \/    =  MZI
             __/\__[]__/\__
    """
    def __init__(self, name_architecture: str,
                 N: int, insersion_loss_MMI: float, imbalance_MMI: float, insersion_loss_Crossing:float, cross_talk_Crossing: float):
        super(Arch_lastMZI_NxN, self).__init__()
        self.N = N
        if name_architecture == 'ClementsBellNxN': self.architecture = ClementsBellNxN(N=N, insersion_loss_MMI=insersion_loss_MMI, imbalance_MMI=imbalance_MMI)
        elif name_architecture == 'FldzhyanBellNxN': self.architecture = FldzhyanBellNxN(N=N, insersion_loss_MMI=insersion_loss_MMI, imbalance_MMI=imbalance_MMI)
        elif name_architecture == 'FldzhyanBellHalfNxN': self.architecture = FldzhyanBellHalfNxN(N=N, insersion_loss_MMI=insersion_loss_MMI, imbalance_MMI=imbalance_MMI)
        elif name_architecture == 'NEUROPULSBonus_unitaryNxN': self.architecture = NEUROPULSBonus_unitaryNxN(N=N, insersion_loss_MMI=insersion_loss_MMI, imbalance_MMI=imbalance_MMI,
                                                                                                             insersion_loss_Crossing=insersion_loss_Crossing, cross_talk_Crossing=cross_talk_Crossing)
        elif name_architecture == 'NEUROPULSBonus_unitary_2long_NxN': self.architecture = NEUROPULSBonus_unitary_2long_NxN(N=N, insersion_loss_MMI=insersion_loss_MMI, imbalance_MMI=imbalance_MMI,
                                                                                                             insersion_loss_Crossing=insersion_loss_Crossing, cross_talk_Crossing=cross_talk_Crossing)
        elif name_architecture == 'NEUROPULSBonus_unitary_2long_NxN': self.architecture = NEUROPULSBonus_unitary_2long_NxN(N=N, insersion_loss_MMI=insersion_loss_MMI, imbalance_MMI=imbalance_MMI,
                                                                                                             insersion_loss_Crossing=insersion_loss_Crossing, cross_talk_Crossing=cross_talk_Crossing)
        else: raise Exception('Wrong name')

        self._layer_MMI_IN = _layer_matrix_MMI_even_IN(N=N, insersion_loss_MMI=insersion_loss_MMI, imbalance_MMI=imbalance_MMI)
        self._layer_ht_full = _layer_matrix_ht_full(N=2*N)
        self._layer_MMI_OUT = _layer_matrix_MMI_even_OUT(N=N, insersion_loss_MMI=insersion_loss_MMI, imbalance_MMI=imbalance_MMI)

    def forward(self):
        id_ini = torch.eye(self.N, requires_grad=False)      # Identity matrix
        arch_matrix = torch.complex(id_ini, torch.zeros_like(id_ini))
        
        arch_matrix = torch.mm(self.architecture(), arch_matrix)
        arch_matrix = torch.mm(self._layer_MMI_IN(), arch_matrix)
        arch_matrix = torch.mm(self._layer_ht_full(), arch_matrix)
        arch_matrix = torch.mm(self._layer_MMI_OUT(), arch_matrix)
        return arch_matrix


# TODO coda intendo fare????
# Creare nuovi modelli con piu losses??? Ma non sono sicuro che riesca a fare le cose fatte meglio
# Crare nuovo algoritmo to training?
# Provare CON MZM normali
# 
# Arch_lastMZI_normal_NxN =========================================================================================
class Arch_lastMZI_normal_NxN(nn.Module):
    r""" Architecture
    Network:
            -----------
        0__|           |__[MZM]__4
           |           |
        1__|           |__[MZM]__5
           | Archeture |
        2__|           |__[MZM]__6
           |           |
        3__|           |__[MZM]__7
            -----------

        with:
            0__  __[]__  __[]__1
               \/      \/       =  MZI
             __/\______/\______
    """
    def __init__(self, name_architecture: str,
                 N: int, insersion_loss_MMI: float, imbalance_MMI: float, insersion_loss_Crossing:float, cross_talk_Crossing: float):
        super(Arch_lastMZI_normal_NxN, self).__init__()
        self.N = N
        if name_architecture == 'ClementsBellNxN': self.architecture = ClementsBellNxN(N=N, insersion_loss_MMI=insersion_loss_MMI, imbalance_MMI=imbalance_MMI)
        elif name_architecture == 'FldzhyanBellNxN': self.architecture = FldzhyanBellNxN(N=N, insersion_loss_MMI=insersion_loss_MMI, imbalance_MMI=imbalance_MMI)
        elif name_architecture == 'FldzhyanBellHalfNxN': self.architecture = FldzhyanBellHalfNxN(N=N, insersion_loss_MMI=insersion_loss_MMI, imbalance_MMI=imbalance_MMI)
        elif name_architecture == 'NEUROPULSBonus_unitaryNxN': self.architecture = NEUROPULSBonus_unitaryNxN(N=N, insersion_loss_MMI=insersion_loss_MMI, imbalance_MMI=imbalance_MMI,
                                                                                                             insersion_loss_Crossing=insersion_loss_Crossing, cross_talk_Crossing=cross_talk_Crossing)
        elif name_architecture == 'NEUROPULSBonus_unitary_2long_NxN': self.architecture = NEUROPULSBonus_unitary_2long_NxN(N=N, insersion_loss_MMI=insersion_loss_MMI, imbalance_MMI=imbalance_MMI,
                                                                                                             insersion_loss_Crossing=insersion_loss_Crossing, cross_talk_Crossing=cross_talk_Crossing)
        elif name_architecture == 'NEUROPULSBonus_unitary_2long_NxN': self.architecture = NEUROPULSBonus_unitary_2long_NxN(N=N, insersion_loss_MMI=insersion_loss_MMI, imbalance_MMI=imbalance_MMI,
                                                                                                             insersion_loss_Crossing=insersion_loss_Crossing, cross_talk_Crossing=cross_talk_Crossing)
        else: raise Exception('Wrong name')

        self._layer_MMI_IN = _layer_matrix_MMI_even_IN(N=N, insersion_loss_MMI=insersion_loss_MMI, imbalance_MMI=imbalance_MMI)
        self._layer_ht_even = _layer_matrix_ht_even(N=2*N)
        self._layer_MMI_OUT = _layer_matrix_MMI_even_OUT(N=N, insersion_loss_MMI=insersion_loss_MMI, imbalance_MMI=imbalance_MMI)
        self._layer_ht_full = _layer_matrix_ht_full(N=N)

    def forward(self):
        id_ini = torch.eye(self.N, requires_grad=False)      # Identity matrix
        arch_matrix = torch.complex(id_ini, torch.zeros_like(id_ini))
        
        arch_matrix = torch.mm(self.architecture(), arch_matrix)
        arch_matrix = torch.mm(self._layer_MMI_IN(), arch_matrix)
        arch_matrix = torch.mm(self._layer_ht_even(), arch_matrix)
        arch_matrix = torch.mm(self._layer_MMI_OUT(), arch_matrix)
        arch_matrix = torch.mm(self._layer_ht_full(), arch_matrix)
        return arch_matrix

