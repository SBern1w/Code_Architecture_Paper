# Photon torch models

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

# from Clements import ClementsNxN_withoutArray

from NEUROPULS import *

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
# torch.cuda.set_device(device)


class nn_NEUROPULS(nn.Module):
    def __init__(self, num_neurons_layer: list = [4, 8, 3]):
        super(nn_NEUROPULS, self).__init__()
        self.in_out_neurons = [num_neurons_layer[0]]
        self.in_out_neurons += [max(num_neurons_layer[ind_n], num_neurons_layer[ind_n-1]) for ind_n in range(1, len(num_neurons_layer))]
        self.in_out_neurons += [num_neurons_layer[-1]]

        self.ph_architectures = nn.ModuleList([NEUROPULSNxN(N=max(num_neurons_layer[ind_n], num_neurons_layer[ind_n-1]))
                                     for ind_n in range(1, len(num_neurons_layer))])
        self.diag_layer = nn.ParameterList([nn.Parameter(torch.rand(n_list, device=device), requires_grad=True) for n_list in num_neurons_layer[1:]])
        self.biases = nn.ParameterList([nn.Parameter(torch.rand(n_list, device=device), requires_grad=True) for n_list in num_neurons_layer[1:]])
    
    def forward(self, x):
        if self.in_out_neurons[0] < self.in_out_neurons[1]:      # add 0 at x keeping important data in the middle
            toadd0 = self.in_out_neurons[1]-self.in_out_neurons[0]
            x = F.pad(x, pad=(toadd0//2, (toadd0+1)//2), mode='constant', value=0)
        
        for i_ph, ph_arch in enumerate(self.ph_architectures):
            x = ph_arch(x)
            
            if self.in_out_neurons[i_ph+1] < self.in_out_neurons[i_ph+2]:
                x = self.diag_layer[i_ph] * x + self.biases[i_ph]
                toadd0 = self.in_out_neurons[i_ph+2]-self.in_out_neurons[i_ph+1]
                x = F.pad(x, pad=(toadd0//2, (toadd0+1)//2), mode='constant', value=0)
            else:         # delete part of the model (NxN) output but keeping the center data part
                mid_min = self.in_out_neurons[i_ph+1]//2 - self.in_out_neurons[i_ph+2]//2
                mid_max = self.in_out_neurons[i_ph+1]//2 + (self.in_out_neurons[i_ph+2]+1)//2
                x = x[:, mid_min:mid_max]
                x = self.diag_layer[i_ph] * x + self.biases[i_ph]
        return x


class Classic_Model(nn.Module):
    name = "Classic_Model"
    n_neuron = 8

    def __init__(self):
        super(Classic_Model, self).__init__()
        self.layer1 = nn.Linear(4, self.n_neuron)
        self.layer2 = nn.Linear(self.n_neuron, self.n_neuron)
        self.layer3 = nn.Linear(self.n_neuron, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x



if __name__ == "__main__":
    print("Test")
    # nn_Clements_DiagWeigthOut([4, 8, 3])

    model = nn_NEUROPULS(num_neurons_layer = [4, 800, 200, 3])

    # # TEST NEUROPULS ----------------------------------------------------------------------------------------------
    # from IRIS_training_backprop import graph_ph_model
    # # HYPERPARAMETERS ---------------------------------------------------------------------------------------------
    # n_epochs = 150
    # batch_size = 12
    # learning_rates = {'diag_layer': 0.008, 'biases': 0.008, 'ph_architectures': 0.0002}
    # model = nn_NEUROPULS(num_neurons_layer = [4, 8, 9, 3])
    
    # graph_ph_model(model, n_epochs, batch_size, learning_rates)
    # # -------------------------------------------------------------------------------------------------------------
    
    
    # # VISUALIZE PARAMETER ----------------------------------------------------
    # model = nn_NEUROPULS(num_neurons_layer = [4, 8, 3])
    # params = list(model.parameters())
    # print(params)

    # for param in params:
    #     print(param.shape)
    # # -------------------------------------------------------------------------



    # VISUALIZE TREE GRADIENT ----------------------------------------------------
    # from torchviz import make_dot

    # input = torch.tensor([[1., 1., 1., 1.],
    #                      [1., 0., 0., 1.],
    #                      [0., 1., 1., 0.]])
    # model = nn_NEUROPULS(num_neurons_layer = [4, 8, 3])

    # output = model(input)
    # loss = torch.sum(output)
    # loss.backward()
    # dot = make_dot(output, params=dict(model.named_parameters()))
    # dot.save("gradient_tree.dot")
    # dot.render(filename='gradient_tree', format='png')
    # dot.format = 'png'
    # dot.render(filename='gradient_tree')
    # -----------------------------------------------------------------------------





