'''
Here are all the most important models, all the last most sure models code will
be reportd in this file.
This code will not work on the GPUs cluster!!

How is structured code:
TRADITIONAL MODELS
    Traditional_NN

OPTICAL ARCHITECTURES
    _layer_wg1
    _layer_wg2
    _layer_MMI1
    _layer_MMI2
    NEUROPULSNxN_2
    NEUROPULSNxN_2_short
    NEUROPULS_2_half
    ClementsNxN

OPTICAL MODEL
    Optical_NN


TODO: add the losses, inbalanced
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
    def __init__(self, name_architecture: str = None, num_neurons_layer: list = [4, 8, 3]):
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
# ========================================== OPTICAL ARCHITECTURES ================================================
# =================================================================================================================

# Layers use for all the optical NN
class _layer_wg1(nn.Module):
    r""" Create waveguide array layer 1
    0__[]__0

    1______1

    2__[]__2
    
    3______3
    """
    def __init__(self, N: int = 2):
        super(_layer_wg1, self).__init__()
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


class _layer_wg2(nn.Module):
    r""" Create waveguide array layer 2
    0______0

    1__[]__1

    2______2
    
    3______3
    """
    def __init__(self, N: int = 2):
        super(_layer_wg2, self).__init__()
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
    

class _layer_wg3(nn.Module):
    r""" Create waveguide array layer 3
    0__[]__0

    1__[]__1

    2__[]__2
    
    3______3
    """
    def __init__(self, N: int = 2):
        super(_layer_wg3, self).__init__()
        self.phase_shift = nn.Parameter(2*torch.pi*torch.rand(N, device=device), requires_grad=True)
        # Waveguide connections
        self.wg_delete = torch.ones(N, requires_grad=False, device=device)  # Delete the waveguide from parameters
        self.add_conns = torch.zeros(N, requires_grad=False, device=device) # Add the connections
        self.wg_delete[-1] = 0.0
        self.add_conns[-1] = 1.0
        
    def forward(self):
        wg_array = torch.exp(1.j*self.phase_shift)
        wg_circuit_matrix = torch.diag(wg_array * self.wg_delete + self.add_conns)
        return wg_circuit_matrix


class _layer_wg0(nn.Module):
    r""" Create waveguide array layer 0
    0__[]__0

    1__[]__1

    2__[]__2
    
    3__[]__3
    """
    def __init__(self, N: int = 2):
        super(_layer_wg0, self).__init__()
        self.phase_shift = nn.Parameter(2*torch.pi*torch.rand(N, device=device), requires_grad=True)
        
    def forward(self):
        wg_array = torch.exp(1.j*self.phase_shift)
        wg_circuit_matrix = torch.diag(wg_array)
        return wg_circuit_matrix


class _layer_MMI1(nn.Module):
    r""" Create MMI array layer 1
    0__  __0
       \/
    1__/\__1
    
    2__  __2
       \/
    3__/\__3
    """
    def __init__(self, N: int, MMI_i_loss_dB: float, MMI_imbal_dB: float):
        super(_layer_MMI1, self).__init__()
        # Loss and imbalances
        i_loss = 10**(-MMI_i_loss_dB/10)
        err_imbal = 1-10**(-MMI_imbal_dB/2/10) if np.random.random() < 0.5 else -(1-10**(-MMI_imbal_dB/2/10))
        err_matrix = torch.tensor([[err_imbal, -err_imbal*1.j],
                                   [err_imbal*1.j, -err_imbal]],
                                   requires_grad=False)
        MMI_device = i_loss*np.sqrt(0.5)*torch.tensor([[1, 1.j],
                                                       [1.j, 1]],
                                                       requires_grad=False)    # 2x2 MMI transfer matrix
        MMI_device = MMI_device + err_matrix
        # MMI connections
        self.MMI_matrix = torch.zeros(size=(N,N), dtype=torch.cfloat, requires_grad=False, device=device)
        for i in range(0, N-1, 2):
            self.MMI_matrix[i:i+2, i:i+2] = MMI_device
        if N%2 == 1:
            self.MMI_matrix[N-1, N-1] = 1.0
    
    def forward(self):
        return self.MMI_matrix


class _layer_MMI2(nn.Module):
    r""" Create MMI array layer 2
    0______0

    1__  __1
       \/
    2__/\__2
    
    3______3
    """
    def __init__(self, N: int, MMI_i_loss_dB: float, MMI_imbal_dB: float):
        super(_layer_MMI2, self).__init__()
        # Loss and imbalances
        i_loss = 10**(-MMI_i_loss_dB/10)
        err_imbal = 1-10**(-MMI_imbal_dB/2/10) if np.random.random() < 0.5 else -(1-10**(-MMI_imbal_dB/2/10))
        err_matrix = torch.tensor([[err_imbal, -err_imbal*1.j],
                                   [err_imbal*1.j, -err_imbal]],
                                   requires_grad=False)
        MMI_device = i_loss*np.sqrt(0.5)*torch.tensor([[1, 1.j],
                                                       [1.j, 1]],
                                                       requires_grad=False)    # 2x2 MMI transfer matrix
        MMI_device = MMI_device + err_matrix
        # MMI connections
        self.MMI_matrix = torch.zeros(size=(N,N), dtype=torch.cfloat, requires_grad=False, device=device)
        self.MMI_matrix[0, 0] = 1.0
        for i in range(1, N-1, 2):
            self.MMI_matrix[i:i+2, i:i+2] = MMI_device
        if N%2 == 0:
            self.MMI_matrix[-1, -1] = 1.0
    
    def forward(self):
        return self.MMI_matrix


class _layer_MMI_IN(nn.Module):
    r""" Create MMI array layer IN
    0__  __0
       \/
       /\__1
    
    2__  __2
       \/
       /\__3
    """
    def __init__(self, N: int):
        super(_layer_MMI_IN, self).__init__()
        MMI_device = np.sqrt(0.5)*torch.tensor([[1], 
                                                [1]],
                                                requires_grad=False)    # 1x2 MMI transfer matrix
        # MMI connections
        self.MMI_matrix = torch.zeros(size=(2*N,N), dtype=torch.cfloat, requires_grad=False, device=device)
        for i in range(0, N, 1):
            self.MMI_matrix[2*i:2*i+2, i:i+1] = MMI_device
        if N%2 == 1:
            self.MMI_matrix[N-1, N-1] = 1.0
    
    def forward(self):
        return self.MMI_matrix


class _layer_MMI_OUT(nn.Module):
    r""" Create MMI array layer OUT
    0__  __0
       \/
    1__/\
    
    2__  __2
       \/
    3__/\
    """
    def __init__(self, N: int):
        super(_layer_MMI_OUT, self).__init__()
        MMI_device = np.sqrt(0.5)*torch.tensor([[1, 1]],
                                                       requires_grad=False)    # 1x2 MMI transfer matrix
        # MMI connections
        self.MMI_matrix = torch.zeros(size=(N,2*N), dtype=torch.cfloat, requires_grad=False, device=device)
        for i in range(0, N, 1):
            self.MMI_matrix[i:i+1, 2*i:2*i+2] = MMI_device
        if N%2 == 1:
            self.MMI_matrix[N-1, N-1] = 1.0
    
    def forward(self):
        return self.MMI_matrix


class _layer_Cross2(nn.Module):
    r""" Create Crossing array layer 2
    0______0

    1__  __1
       \/
    2__/\__2
    
    3______3
    """
    def __init__(self, N: int):
        super(_layer_Cross2, self).__init__()
        MMI_device = torch.tensor([[0, 1],
                                   [1, 0]],
                                   requires_grad=False)    # 2x2 Cross
        # MMI connections
        self.MMI_matrix = torch.zeros(size=(N,N), dtype=torch.cfloat, requires_grad=False, device=device)
        self.MMI_matrix[0, 0] = 1.0
        for i in range(1, N-1, 2):
            self.MMI_matrix[i:i+2, i:i+2] = MMI_device
        if N%2 == 0:
            self.MMI_matrix[-1, -1] = 1.0
    
    def forward(self):
        return self.MMI_matrix


# NEUROPULS_2 =====================================================================================================
class NEUROPULSNxN_2(nn.Module):
    r""" NEUROPULS_2 architecture
    Network 4x4:
        <---------------------------2N------------------------------------> Length number of layer MMIs
        0__[]__  __________[]__  __________[]__  __________[]__  __________[]__0
               \/              \/              \/              \/
        1______/\__[]__  ______/\__[]__  ______/\__[]__  ______/\__[]__  __[]__1
                       \/              \/              \/              \/
        2__[]__  ______/\__[]__  ______/\__[]__  ______/\__[]__  ______/\__[]__2
               \/              \/              \/              \/
        3______/\______________/\______________/\______________/\__________[]__3

        with:
            0__[]__1 = phase shift

            3__  __2
               \/    =  MMI
            0__/\__1
    """
    def __init__(self, N: int, MMI_i_loss_dB: float, MMI_imbal_dB: float):
        super(NEUROPULSNxN_2, self).__init__()
        self.N = N
        self.wg1_layer = nn.ModuleList([_layer_wg1(N=N) for i in range(N)])
        self.wg2_layer = nn.ModuleList([_layer_wg2(N=N) for i in range(N)])
        self.MMI1_layer = nn.ModuleList([_layer_MMI1(N=N,
                                                     MMI_i_loss_dB=MMI_i_loss_dB,
                                                     MMI_imbal_dB=MMI_imbal_dB) for i in range(N)])
        self.MMI2_layer = nn.ModuleList([_layer_MMI2(N=N,
                                                     MMI_i_loss_dB=MMI_i_loss_dB,
                                                     MMI_imbal_dB=MMI_imbal_dB) for i in range(N)])
        self.wg0_layer = _layer_wg0(N=N)

    def forward(self):
        id_ini = torch.eye(self.N)
        arch_matrix = torch.complex(id_ini, torch.zeros_like(id_ini))
        for i in range(self.N):
            arch_matrix = torch.mm(self.wg1_layer[i](), arch_matrix)
            arch_matrix = torch.mm(self.MMI1_layer[i](), arch_matrix)
            arch_matrix = torch.mm(self.wg2_layer[i](), arch_matrix)
            arch_matrix = torch.mm(self.MMI2_layer[i](), arch_matrix)
        arch_matrix = torch.mm(self.wg0_layer(), arch_matrix)
        return arch_matrix


# Bell =====================================================================================================
class BellNxN(nn.Module):
    r""" BellNxN architecture
    Network 4x4:
        <---------------------------2N------------------------------------> Length number of layer MMIs
        0__[]__  __________[]__  __________[]__  __________[]__  __________[]__0
               \/              \/              \/              \/
        1__[]__/\______  __[]__/\______  __[]__/\______  __[]__/\______  __[]__1
                       \/              \/              \/              \/
        2__[]__  ______/\__[]__  ______/\__[]__  ______/\__[]__  ______/\__[]__2
               \/              \/              \/              \/
        3__[]__/\__________[]__/\__________[]__/\__________[]__/\__________[]__3

        with:
            0__[]__1 = phase shift

            3__  __2
               \/    =  MMI
            0__/\__1
    """
    def __init__(self, N: int, MMI_i_loss_dB: float, MMI_imbal_dB: float):
        super(BellNxN, self).__init__()
        self.N = N
        self.wg_layer = nn.ModuleList([_layer_wg0(N=N) for i in range(N)])
        self.MMI1_layer = nn.ModuleList([_layer_MMI1(N=N,
                                                     MMI_i_loss_dB=MMI_i_loss_dB,
                                                     MMI_imbal_dB=MMI_imbal_dB) for i in range(N)])
        self.MMI2_layer = nn.ModuleList([_layer_MMI2(N=N,
                                                     MMI_i_loss_dB=MMI_i_loss_dB,
                                                     MMI_imbal_dB=MMI_imbal_dB) for i in range(N)])
        self.wgOUT_layer = _layer_wg0(N=N)

    def forward(self):
        id_ini = torch.eye(self.N)
        arch_matrix = torch.complex(id_ini, torch.zeros_like(id_ini))
        for i in range(self.N):
            arch_matrix = torch.mm(self.wg_layer[i](), arch_matrix)
            arch_matrix = torch.mm(self.MMI1_layer[i](), arch_matrix)
            arch_matrix = torch.mm(self.MMI2_layer[i](), arch_matrix)
        arch_matrix = torch.mm(self.wgOUT_layer(), arch_matrix)
        return arch_matrix



# BonusIdea =====================================================================================================
class BonusIdeaNxN(nn.Module):
    r""" BonusIdeaNxN architecture
    Network 4x4:

        with:
            0__[]__1 = phase shift

            3__  __2
               \/    =  MMI
            0__/\__1
    """
    def __init__(self, N: int, MMI_i_loss_dB: float, MMI_imbal_dB: float):
        super(BonusIdeaNxN, self).__init__()
        self.N = N
        self.MMI_IN_layer = _layer_MMI_IN(N=N)
        self.wg_IN_layer = _layer_wg0(N=N)
        self.wg1_layer = nn.ModuleList([_layer_wg0(N=2*N) for i in range(2*N)])
        self.MMI1_layer = nn.ModuleList([_layer_MMI1(N=2*N,
                                                     MMI_i_loss_dB=MMI_i_loss_dB,
                                                     MMI_imbal_dB=MMI_imbal_dB) for i in range(2*N)])
        self.MMI_OUT_layer = _layer_MMI_OUT(N=N)
        self.Cross = nn.ModuleList([_layer_Cross2(N=2*N) for i in range(N-1)])

    def forward(self):
        arch_matrix = self.wg_IN_layer()
        arch_matrix = torch.mm(self.MMI_IN_layer(), arch_matrix)
        for i in range(self.N):
            if i > 0:
                arch_matrix = torch.mm(self.wg1_layer[2*i](), arch_matrix)
                arch_matrix = torch.mm(self.MMI1_layer[2*i](), arch_matrix)
            arch_matrix = torch.mm(self.wg1_layer[2*i+1](), arch_matrix)
            if i < self.N-1:
                arch_matrix = torch.mm(self.MMI1_layer[2*i+1](), arch_matrix)
                arch_matrix = torch.mm(self.Cross[i](), arch_matrix)
            else:
                arch_matrix = torch.mm(self.MMI_OUT_layer(), arch_matrix)
        return arch_matrix
    


# BonusIdea_2 =====================================================================================================
class BonusIdea_2_NxN(nn.Module):
    r""" BonusIdea_2_NxN architecture
    Network 4x4:

        with:
            0__[]__1 = phase shift

            3__  __2
               \/    =  MMI
            0__/\__1
    """
    def __init__(self, N: int, MMI_i_loss_dB: float, MMI_imbal_dB: float):
        super(BonusIdea_2_NxN, self).__init__()
        self.N = N
        self.wg0_layer = nn.ModuleList([_layer_wg0(N=N) for i in range(N+1)])
        self.MMI1_layer = nn.ModuleList([_layer_MMI1(N=N,
                                                     MMI_i_loss_dB=MMI_i_loss_dB,
                                                     MMI_imbal_dB=MMI_imbal_dB) for i in range(N)])
        self.Cross = nn.ModuleList([_layer_Cross2(N=N) for i in range(N-1)])

    def forward(self):
        id_ini = torch.eye(self.N)
        arch_matrix = torch.complex(id_ini, torch.zeros_like(id_ini))
        for i in range(self.N):
            arch_matrix = torch.mm(self.wg0_layer[i](), arch_matrix)
            arch_matrix = torch.mm(self.MMI1_layer[i](), arch_matrix)
            if i < self.N-1:
                arch_matrix = torch.mm(self.Cross[i](), arch_matrix)
        
        arch_matrix = torch.mm(self.wg0_layer[-1](), arch_matrix)
        return arch_matrix




# Clements ========================================================================================================
class ClementsNxN(nn.Module):
    r""" Clements architecture
    Network:
        <-------------------------------2N--------------------------------> Length number of layer MMIs
        0__[]__  __[]__  __________________[]__  __[]__  __________________[]__4
               \/      \/                      \/      \/
        1______/\______/\__[]__  __[]__  ______/\______/\__[]__  __[]__  __[]__5
                               \/      \/                      \/      \/
        2__[]__  __[]__  ______/\______/\__[]__  __[]__  ______/\______/\__[]__6
               \/      \/                      \/      \/
        3______/\______/\______________________/\______/\__________________[]__7

        with:
            0__[]__1 = phase shift

            3__  __2
               \/    =  MMI
            0__/\__1
    """
    def __init__(self, N: int, MMI_i_loss_dB: float, MMI_imbal_dB: float):
        super(ClementsNxN, self).__init__()
        self.N = N
        if N%2 == 0:
            self.wg1_layer = nn.ModuleList([_layer_wg1(N=N) for i in range(N)])
            self.wg2_layer = nn.ModuleList([_layer_wg2(N=N) for i in range(N)])
            self.MMI1_layer = nn.ModuleList([_layer_MMI1(N=N,
                                                        MMI_i_loss_dB=MMI_i_loss_dB,
                                                        MMI_imbal_dB=MMI_imbal_dB) for i in range(N)])
            self.MMI2_layer = nn.ModuleList([_layer_MMI2(N=N,
                                                        MMI_i_loss_dB=MMI_i_loss_dB,
                                                        MMI_imbal_dB=MMI_imbal_dB) for i in range(N)])
        else:
            self.wg1_layer = nn.ModuleList([_layer_wg1(N=N) for i in range(N+1)])
            self.wg2_layer = nn.ModuleList([_layer_wg2(N=N) for i in range(N-1)])
            self.MMI1_layer = nn.ModuleList([_layer_MMI1(N=N,
                                                        MMI_i_loss_dB=MMI_i_loss_dB,
                                                        MMI_imbal_dB=MMI_imbal_dB) for i in range(N+1)])
            self.MMI2_layer = nn.ModuleList([_layer_MMI2(N=N,
                                                        MMI_i_loss_dB=MMI_i_loss_dB,
                                                        MMI_imbal_dB=MMI_imbal_dB) for i in range(N-1)])
        self.wg0_layer = _layer_wg0(N=N)


    def forward(self):
        id_ini = torch.eye(self.N)
        arch_matrix = torch.complex(id_ini, torch.zeros_like(id_ini))
        for i in range(self.N//2):
            arch_matrix = torch.mm(self.wg1_layer[2*i](), arch_matrix)
            arch_matrix = torch.mm(self.MMI1_layer[2*i](), arch_matrix)
            arch_matrix = torch.mm(self.wg1_layer[2*i+1](), arch_matrix)
            arch_matrix = torch.mm(self.MMI1_layer[2*i+1](), arch_matrix)
            arch_matrix = torch.mm(self.wg2_layer[2*i](), arch_matrix)
            arch_matrix = torch.mm(self.MMI2_layer[2*i](), arch_matrix)
            arch_matrix = torch.mm(self.wg2_layer[2*i+1](), arch_matrix)
            arch_matrix = torch.mm(self.MMI2_layer[2*i+1](), arch_matrix)
        i += 1
        if self.N%2 == 1:
            arch_matrix = torch.mm(self.wg1_layer[2*i](), arch_matrix)
            arch_matrix = torch.mm(self.MMI1_layer[2*i](), arch_matrix)
            arch_matrix = torch.mm(self.wg1_layer[2*i+1](), arch_matrix)
            arch_matrix = torch.mm(self.MMI1_layer[2*i+1](), arch_matrix)

        arch_matrix = torch.mm(self.wg0_layer(), arch_matrix)
        return arch_matrix




# # TEST singular module ------------------------------------------------------------------------------------------
# N=8
# module = NEUROPULSNxN(N=N)
# input = torch.ones(N)
# module(torch.tensor(input))
# # module(torch.complex(input, torch.zeros_like(input)))
# # ---------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Test")

    # Slow but run good
    # model = NEUROPULSNxN(N=7)


    # # VISUALIZE PARAMETER ---------------------------------------------------------------------------------------
    # model = NEUROPULSNxN(N=5)
    # # model = Classic_Model()
    # params = list(model.parameters())
    # print(params)

    # for param in params:
    #     print(param.shape)
    # # -----------------------------------------------------------------------------------------------------------

    
    # VISUALIZE TREE GRADIENT -------------------------------------------------------------------------------------
    # from torchviz import make_dot

    # input = torch.tensor([1., 1., 1., 1.])
    # model = NEUROPULSNxN_test(N=4)

    # output = model(input)
    # loss = torch.sum(output)
    # loss.backward()
    # dot = make_dot(output, params=dict(model.named_parameters()))
    # dot.save("gradient_tree.dot")
    # dot.render(filename='gradient_tree', format='png')
    # dot.format = 'png'
    # dot.render(filename='gradient_tree')
    # -------------------------------------------------------------------------------------------------------------



