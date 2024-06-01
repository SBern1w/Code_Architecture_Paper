'''
All the models

Traditional

Optical

'''


import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

# This is commands are used for the GPUS. FOr this code, I will use just the CPU!!!
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   # the fucntion gives errors sometimes
# torch.cuda.set_device(device)
device = "cpu"

# =================================================================================================================
# ========================================== TRADITIONAL MODELS ===================================================
# =================================================================================================================

# Traditional NN ==================================================================================================
class Traditional_NN(nn.Module):
    def __init__(self, name_architecture: str, num_neurons_layer: list):
        super(Traditional_NN, self).__init__()
        n_layers = len(num_neurons_layer)
        self.layers = nn.ModuleList([nn.Linear(num_neurons_layer[i-1], num_neurons_layer[i], bias=True)
                                     for i in range(1, n_layers)])
        # Activations
        self.activeReLU = nn.ReLU()
        self.activeSoftMax = nn.Softmax(dim=1)

    def forward(self, x):
        for ind_l, layer in enumerate(self.layers):
            x = layer(x)
            if not ind_l == len(self.layers) - 1:
                x = self.activeReLU(x)
        x = self.activeSoftMax(x)
        return x



# =================================================================================================================
# ========================================== OPTICAL LAYERS for architectures =====================================
# =================================================================================================================

class _layer_matrix_ht_full(nn.Module):
    r""" Create waveguide matrix layer with all the line connected to a simple ht
    0__[]__0

    1__[]__1

    2__[]__2
    
    3__[]__3
    """
    def __init__(self, N: int):
        super(_layer_matrix_ht_full, self).__init__()
        self.phase_shift = nn.Parameter(2*torch.pi*torch.rand(N, device=device), requires_grad=True)
    
    def forward(self):
        wg_array = torch.exp(1.j*self.phase_shift)
        wg_circuit_matrix = torch.diag(wg_array)
        return wg_circuit_matrix


class _layer_matrix_ht_even(nn.Module):
    r""" Create waveguide matrix layer with just even lines connected to a simple ht
    0__[]__0

    1______1

    2__[]__2
    
    3______3
    """
    def __init__(self, N: int = 2):
        super(_layer_matrix_ht_even, self).__init__()
        self.phase_shift = nn.Parameter(2*torch.pi*torch.rand(N, device=device), requires_grad=True)
        # Waveguide connections
        self.wg_delete = torch.ones(N, requires_grad=False, device=device)  # Delete the waveguide from parameters
        self.add_conns = torch.zeros(N, requires_grad=False, device=device) # Add the connections
        for i in range(1, N, 2):
            self.wg_delete[i] = 0.0
            self.add_conns[i] = 1.0
        if N%2 == 1:
            self.wg_delete[-1] = 0.0
            self.add_conns[-1] = 1.0
    
    def forward(self):
        wg_array = torch.exp(1.j*self.phase_shift)
        wg_circuit_matrix = torch.diag(wg_array * self.wg_delete + self.add_conns)
        return wg_circuit_matrix


class _layer_matrix_ht_odd(nn.Module):
    r""" Create waveguide matrix layer with just odd lines connected to a simple ht
    0______0

    1__[]__1

    2______2
    
    3______3
    """
    def __init__(self, N: int = 2):
        super(_layer_matrix_ht_odd, self).__init__()
        self.phase_shift = nn.Parameter(2*torch.pi*torch.rand(N, device=device), requires_grad=True)
        # Waveguide connections
        self.wg_delete = torch.ones(N, requires_grad=False, device=device)  # Delete the waveguide from parameters
        self.add_conns = torch.zeros(N, requires_grad=False, device=device) # Add the connections
        for i in range(0, N, 2):
            self.wg_delete[i] = 0.0
            self.add_conns[i] = 1.0
        if N%2 == 0:
            self.wg_delete[-1] = 0.0
            self.add_conns[-1] = 1.0
        
    def forward(self):
        wg_array = torch.exp(1.j*self.phase_shift)
        wg_circuit_matrix = torch.diag(wg_array * self.wg_delete + self.add_conns)
        return wg_circuit_matrix


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


class _layer_matrix_MMI_even(nn.Module):
    r""" Create a MMI matrix with first pin connect to even imput
    0__  __0
       \/
    1__/\__1
    
    2__  __2
       \/
    3__/\__3
    """
    def __init__(self, N: int, insersion_loss_MMI: float, imbalance_MMI: float):
        super(_layer_matrix_MMI_even, self).__init__()
        # 2x2 MMI transfer matrix with insersion loss and imbalance
        MMI_device = np.sqrt(1-insersion_loss_MMI)*torch.tensor([[np.sqrt(1/2+imbalance_MMI), 1.j*np.sqrt(1/2-imbalance_MMI)],
                                                                 [1.j*np.sqrt(1/2-imbalance_MMI), np.sqrt(1/2+imbalance_MMI)]],
                                                                 requires_grad=False)
        # MMI connections
        self.MMI_matrix = torch.zeros(size=(N,N), dtype=torch.cfloat, requires_grad=False, device=device)
        for i in range(0, N-1, 2):
            self.MMI_matrix[i:i+2, i:i+2] = MMI_device
        if N%2 == 1:
            self.MMI_matrix[N-1, N-1] = 1.0
    
    def forward(self):
        return self.MMI_matrix


class _layer_matrix_MMI_odd(nn.Module):
    r""" Create a MMI matrix with first pin connect to odd imput
    0______0

    1__  __1
       \/
    2__/\__2
    
    3______3
    """
    def __init__(self, N: int, insersion_loss_MMI: float, imbalance_MMI: float):
        super(_layer_matrix_MMI_odd, self).__init__()
        # 2x2 MMI transfer matrix with insersion loss and imbalance
        MMI_device = np.sqrt(1-insersion_loss_MMI)*torch.tensor([[np.sqrt(1/2+imbalance_MMI), 1.j*np.sqrt(1/2-imbalance_MMI)],
                                                                 [1.j*np.sqrt(1/2-imbalance_MMI), np.sqrt(1/2+imbalance_MMI)]],
                                                                 requires_grad=False)
        # MMI connections
        self.MMI_matrix = torch.zeros(size=(N,N), dtype=torch.cfloat, requires_grad=False, device=device)
        self.MMI_matrix[0, 0] = 1.0
        for i in range(1, N-1, 2):
            self.MMI_matrix[i:i+2, i:i+2] = MMI_device
        if N%2 == 0:
            self.MMI_matrix[-1, -1] = 1.0
    
    def forward(self):
        return self.MMI_matrix


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


# ClementsBell ====================================================================================================
class ClementsBellNxN(nn.Module):
    r""" Clements Bell architecture
    Network:
        <---------------------------------2N---------------------------------> Length number of layer MMIs
        0__[]__  __[]__  __________[]__________  __[]__  __________[]______[]__4
               \/      \/                      \/      \/
        1__[]__/\__[]__/\______  __[]__  ______/\__[]__/\______  __[]__  __[]__5
                               \/      \/                      \/      \/
        2__[]__  __[]__  ______/\__[]__/\______  __[]__  ______/\__[]__/\__[]__6
               \/      \/                      \/      \/
        3__[]__/\__[]__/\__________[]__________/\__[]__/\__________[]______[]__7

        with:
            0__[]__1 = phase shift

            3__  __2
               \/    =  MMI
            0__/\__1
    """
    def __init__(self, N: int, insersion_loss_MMI: float, imbalance_MMI: float):
        super(ClementsBellNxN, self).__init__()
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
        for i in range(self.N//2):
            arch_matrix = torch.mm(self._layer_MMI_even[2*i](), arch_matrix)
            arch_matrix = torch.mm(self._layer_ht[2*i+1](), arch_matrix)
            arch_matrix = torch.mm(self._layer_MMI_even[2*i+1](), arch_matrix)
            arch_matrix = torch.mm(self._layer_MMI_odd[2*i](), arch_matrix)
            arch_matrix = torch.mm(self._layer_ht[2*i+2](), arch_matrix)
            arch_matrix = torch.mm(self._layer_MMI_odd[2*i+1](), arch_matrix)
        arch_matrix = torch.mm(self._layer_ht[2*i+3](), arch_matrix)
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
            3__  __[]__  __2
               \/      \/    =  MZI
            0__/\__[]__/\__1
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



