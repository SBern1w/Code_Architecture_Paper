# Tracking the random target matrix

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import unitary_group

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm         # make loops show as a smart progress meter

from matrix_models import *


# HIGH PARAMETERS =================================================================================================
seed = 37

Ns = [4, 8]

lr = 0.001
# momentum = 0.9

n_epochs_xN = [20000]

n_matrices = 20
n_repetitions = 1


# name_models = ['ClementsBellNxN', 'FldzhyanBellNxN', 'FldzhyanBellHalfNxN', 'NEUROPULSBonus_unitaryNxN', 'NEUROPULSBonus_unitary_2long_NxN',
#                'NEUROPULSBonus_anymatrixNxN', 'NEUROPULSBonus_Bell_Minht_NxN', 'NEUROPULSBonus_BellNormal_NxN', 'CB_withMZI']
# is_unitarys = [True, True, True, True, True,
#               False, False, False,]

name_model = 'CB_withMZI'
is_unitary = False

# folder_path = '20240403_matrix_optimization/outdata/20240409_data/'
folder_path = 'outdata/20240412_data/'
# =================================================================================================================

torch.manual_seed(seed)         # Seed for the random number
np.random.seed(seed)

# Compless MSE Loss function, basically the definition
class CMatrixMSELoss(nn.Module):
    def __init__(self):
        super(CMatrixMSELoss, self).__init__()

    def forward(self, predicted_matrix, target_matrix):
        mag_diff_sq = torch.abs(predicted_matrix - target_matrix)**2
        loss = torch.sum(mag_diff_sq) / torch.numel(target_matrix)
        return loss

for i_N, N in enumerate(Ns):
    n_epochs = n_epochs_xN[i_N]

    matrix_target_predictions = []
    loss_xepoch = []
    for i_matrix in range(n_matrices):
        if is_unitary:
            # UNITARY MATRIX
            target_matrix = torch.tensor(unitary_group.rvs(N))    # Unitary matrix
        else:
            # UNIVERSAL MATRIX
            real_part = torch.randn(N, N)
            imag_part = torch.randn(N, N)
            complex_matrix = torch.complex(real_part, imag_part)
            column_norms = torch.sum((torch.abs(complex_matrix))**2, dim=0)     # Power outputs
            scaling_factor = 0.5 * torch.rand(1) / torch.sqrt(torch.max(column_norms)/2)        # Random scaling between 0.5 and 0!!!!!!!!!!!!!
            target_matrix = complex_matrix * scaling_factor
            # print(torch.sum((torch.abs(target_matrix))**2, dim=0))

        matrix_target_predictions.append(('target'+'_i_matrix'+str(i_matrix), target_matrix.numpy()))
        # end_rep = False
        for rep in range(n_repetitions):        # Repetions
            # UNITARY MATRIX
            if name_model == 'ClementsBellNxN': model = ClementsBellNxN(N=N, insersion_loss_MMI=0., imbalance_MMI=0.)
            elif name_model == 'FldzhyanBellNxN': model = FldzhyanBellNxN(N=N, insersion_loss_MMI=0., imbalance_MMI=0.)
            elif name_model == 'FldzhyanBellHalfNxN': model = FldzhyanBellHalfNxN(N=N, insersion_loss_MMI=0., imbalance_MMI=0.)
            elif name_model == 'NEUROPULSBonus_unitaryNxN': model = NEUROPULSBonus_unitaryNxN(N=N, insersion_loss_MMI=0., imbalance_MMI=0., insersion_loss_Crossing=0., cross_talk_Crossing=0.)
            elif name_model == 'NEUROPULSBonus_unitary_2long_NxN': model = NEUROPULSBonus_unitary_2long_NxN(N=N, insersion_loss_MMI=0., imbalance_MMI=0., insersion_loss_Crossing=0., cross_talk_Crossing=0.)
            # UNIVERSAL MATRIX
            elif name_model == 'NEUROPULSBonus_anymatrixNxN': model = NEUROPULSBonus_anymatrixNxN(N=N, insersion_loss_MMI=0., imbalance_MMI=0., insersion_loss_Crossing=0., cross_talk_Crossing=0.)
            elif name_model == 'NEUROPULSBonus_Bell_Minht_NxN': model = NEUROPULSBonus_Bell_Minht_NxN(N=N, insersion_loss_MMI=0., imbalance_MMI=0., insersion_loss_Crossing=0., cross_talk_Crossing=0.)
            elif name_model == 'NEUROPULSBonus_BellNormal_NxN': model = NEUROPULSBonus_BellNormal_NxN(N=N, insersion_loss_MMI=0., imbalance_MMI=0., insersion_loss_Crossing=0., cross_talk_Crossing=0.)
            elif name_model == 'CB_withMZI': model = Arch_lastMZI_NxN(name_architecture='ClementsBellNxN', N=N, insersion_loss_MMI=0., imbalance_MMI=0., insersion_loss_Crossing=0., cross_talk_Crossing=0.)


            loss_fn = CMatrixMSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)                # To tune lr and weight_decay to make loss more stable !!!!!!
            # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

            loss_xepoch_data = []
            for epoch in tqdm.trange(n_epochs):     # Optimiziation with gradient
                prediction_matrix = model()
                loss = loss_fn(prediction_matrix, target_matrix)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_xepoch_data.append(loss.detach().numpy())
                # if loss_xepoch_data[epoch] < 1E-6:                 # If this loss I'm ok let pass next one
                #     end_rep = True
                #     break
            
            print(f'Epoch [{epoch+1}/{n_epochs}], i_matrix: {i_matrix}, model: {name_model}, rep: {rep}, Loss: {loss.item():.10f}')

            matrix_target_predictions.append(('predition'+'_i_matrix'+str(i_matrix)+'_rep'+str(rep), prediction_matrix.detach().numpy()))
            loss_xepoch.append(('loss_xepoch'+'_i_matrix'+str(i_matrix)+'_rep'+str(rep), np.array(loss_xepoch_data)))

            # if end_rep:    # If I found this matrix can be optimize lets go to another matrix
            #     break

    matrix_target_predictions = np.array(matrix_target_predictions, dtype=[('label', 'U100'), ('matrix', 'O')])
    np.save(folder_path+'model_'+name_model+'_matrix_N'+str(N), matrix_target_predictions)
    loss_xepoch = np.array(loss_xepoch, dtype=[('label', 'U100'), ('matrix', 'O')])
    np.save(folder_path+'model_'+name_model+'_loss_N'+str(N), loss_xepoch)






    



