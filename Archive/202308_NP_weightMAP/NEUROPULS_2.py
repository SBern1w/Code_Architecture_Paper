# Create the model of NEUROPULS

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import tqdm         # make loops show as a smart progress meter


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
# torch.cuda.set_device(device)


class _NEUROPULS_layer_wg1(nn.Module):
    r""" Create NEUROPULS waveguide layer 1
    0__[]__0

    1______1

    2__[]__2
    
    3______3
    """
    def __init__(self, N: int = 3):
        super(_NEUROPULS_layer_wg1, self).__init__()
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
    
    def forward(self, x):
        wg_array = torch.exp(1.j*self.phase_shift)
        wg_circuit_matrix = torch.diag(wg_array * self.wg_delete + self.add_conns)
        return F.linear(x, wg_circuit_matrix)


class _NEUROPULS_layer_wg2(nn.Module):
    r""" Create NEUROPULS waveguide layer 2
    0______0

    1__[]__1

    2______2
    
    3______3
    """
    def __init__(self, N: int = 3):
        super(_NEUROPULS_layer_wg2, self).__init__()
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
        
    def forward(self, x):
        wg_array = torch.exp(1.j*self.phase_shift)
        wg_circuit_matrix = torch.diag(wg_array * self.wg_delete + self.add_conns)
        return F.linear(x, wg_circuit_matrix)


class _NEUROPULS_layer_MMI1(nn.Module):
    r""" Create NEUROPULS MMI layer 1
    0__  __0
       \/
    1__/\__1
    
    2__  __2
       \/
    3__/\__3
    """
    def __init__(self, N: int = 3):
        super(_NEUROPULS_layer_MMI1, self).__init__()
        MMI_device = np.sqrt(0.5)*torch.tensor([[1., 1.j],
                                                [1.j, 1.]],
                                                requires_grad=False)
        # MMI connections
        self.MMI_matrix = torch.zeros(size=(N,N), dtype=torch.cfloat, requires_grad=False, device=device)
        for i in range(0, N-1, 2):
            self.MMI_matrix[i:i+2, i:i+2] = MMI_device
        if N%2 == 1:
            self.MMI_matrix[N-1, N-1] = 1.0
    
    def forward(self, x):
        return F.linear(x, self.MMI_matrix)     # Is right => y=xA^T+b


class _NEUROPULS_layer_MMI2(nn.Module):
    r""" Create NEUROPULS MMI layer 2
    0______0

    1__  __1
       \/
    2__/\__2
    
    3______3
    """
    def __init__(self, N: int = 3):
        super(_NEUROPULS_layer_MMI2, self).__init__()
        MMI_device = np.sqrt(0.5)*torch.tensor([[1., 1.j],
                                                [1.j, 1.]],
                                                requires_grad=False)
        # MMI connections
        self.MMI_matrix = torch.zeros(size=(N,N), dtype=torch.cfloat, requires_grad=False, device=device)
        self.MMI_matrix[0, 0] = 1.0
        for i in range(1, N-1, 2):
            self.MMI_matrix[i:i+2, i:i+2] = MMI_device
        if N%2 == 0:
            self.MMI_matrix[-1, -1] = 1.0
    
    def forward(self, x):
        return F.linear(x, self.MMI_matrix)


class NEUROPULSNxN_2_2(nn.Module):
    r""" A approsimate unitary matrix network based on the NEUROPULS architecture.
    Network::

        0__[]__  __________[]__  __________[]__  __________[]__  __________4
               \/              \/              \/              \/
        1______/\__[]__  ______/\__[]__  ______/\__[]__  ______/\__[]__  __5
                       \/              \/              \/              \/
        2__[]__  ______/\__[]__  ______/\__[]__  ______/\__[]__  ______/\__6
               \/              \/              \/              \/
        3______/\______________/\______________/\______________/\__________7

        with:
            0__[]__1 = phase shift

            3__  __2
               \/    =  MMI
            0__/\__1
    """
    def __init__(self, N: int = 3):
        super(NEUROPULSNxN_2_2, self).__init__()
        self.N = N
        self.wg1_layer = nn.ModuleList([_NEUROPULS_layer_wg1(N=N) for i in range(N)])   
        self.wg2_layer = nn.ModuleList([_NEUROPULS_layer_wg2(N=N) for i in range(N)])
        self.MMI1_layer = nn.ModuleList([_NEUROPULS_layer_MMI1(N=N) for i in range(N)])
        self.MMI2_layer = nn.ModuleList([_NEUROPULS_layer_MMI2(N=N) for i in range(N)])

    def forward(self, x):
        x_complex = torch.complex(x, torch.zeros_like(x))
        # Circuit
        for i in range(self.N):
            x_complex = self.wg1_layer[i](x_complex)
            x_complex = self.MMI1_layer[i](x_complex)
            x_complex = self.wg2_layer[i](x_complex)
            x_complex = self.MMI2_layer[i](x_complex)
        # DETECTOR
        real_squared_part = torch.square(torch.real(x_complex))
        imaginary_squared_part = torch.square(torch.imag(x_complex))
        x_detector = real_squared_part + imaginary_squared_part                    # not really really sure
        return x_detector


class NEUROPULSNxN_2_3(nn.Module):
    r""" A approsimate unitary matrix network based on the NEUROPULS architecture.
        Jsut the half of the privious desing
    Network::

        0__[]__  __________[]__  __________4
               \/              \/
        1______/\__[]__  ______/\__[]__  __5
                       \/              \/
        2__[]__  ______/\__[]__  ______/\__6
               \/              \/
        3______/\______________/\__________7

        with:
            0__[]__1 = phase shift

            3__  __2
               \/    =  MMI
            0__/\__1
    """
    def __init__(self, N: int = 3):
        super(NEUROPULSNxN_2_3, self).__init__()
        self.N = N
        self.wg1_layer = nn.ModuleList([_NEUROPULS_layer_wg1(N=N) for i in range(N)])   
        self.wg2_layer = nn.ModuleList([_NEUROPULS_layer_wg2(N=N) for i in range(N)])
        self.MMI1_layer = nn.ModuleList([_NEUROPULS_layer_MMI1(N=N) for i in range(N)])
        self.MMI2_layer = nn.ModuleList([_NEUROPULS_layer_MMI2(N=N) for i in range(N)])

    def forward(self, x):
        x_complex = torch.complex(x, torch.zeros_like(x))
        # Circuit
        for i in range(self.N//2):
            x_complex = self.wg1_layer[i](x_complex)
            x_complex = self.MMI1_layer[i](x_complex)
            x_complex = self.wg2_layer[i](x_complex)
            x_complex = self.MMI2_layer[i](x_complex)
        # DETECTOR
        real_squared_part = torch.square(torch.real(x_complex))
        imaginary_squared_part = torch.square(torch.imag(x_complex))
        x_detector = real_squared_part + imaginary_squared_part                    # not really really sure
        return x_detector


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



