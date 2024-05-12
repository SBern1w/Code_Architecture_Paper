# Tracking the random target matrix

from scoop import futures
# import multiprocessing
import os
import numpy as np
from scipy.stats import unitary_group

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm         # make loops show as a smart progress meter

from matrix_models import *


# HIGH PARAMETERS =================================================================================================
seed = 37

# Hyparameters of the losses
lr = 0.001
n_epochs = 20000

name_models = ['ClementsBellNxN', 'FldzhyanBellNxN', 'FldzhyanBellHalfNxN',] # 'NEUROPULSBonus_unitaryNxN', 'NEUROPULSBonus_unitary_2long_NxN',]
            #    'NEUROPULSBonus_anymatrixNxN', 'NEUROPULSBonus_Bell_Minht_NxN', 'NEUROPULSBonus_BellNormal_NxN',
            #    'CB_withMZI', 'FB_withMZI', 'FBH_withMZI', 'NPBU_withMZI',]
# name_models = ['ClementsBellNxN', 'FldzhyanBellNxN', 'FldzhyanBellHalfNxN', 'NEUROPULSBonus_unitaryNxN', 'NEUROPULSBonus_unitary_2long_NxN',
#                'NEUROPULSBonus_anymatrixNxN', 'NEUROPULSBonus_Bell_Minht_NxN', 'NEUROPULSBonus_BellNormal_NxN', 'CB_withMZI']

Ns = [4,]
n_matrices = 1000
n_repetitions = 5

# folder_path = '20240403_matrix_optimization/outdata/20240409_data/'
folder_path = '20240403_matrix_optimization/outdata/'
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


def model_prediction(args):
    name_model, N, i_matrix, rep, target_matrix_unitary, target_any_matrix = args

    # UNITARY MATRIX
    if name_model == 'ClementsBellNxN':
        model = ClementsBellNxN(N=N, insersion_loss_MMI=0., imbalance_MMI=0.)
        target_matrix = target_matrix_unitary
    elif name_model == 'FldzhyanBellNxN':
        model = FldzhyanBellNxN(N=N, insersion_loss_MMI=0., imbalance_MMI=0.)
        target_matrix = target_matrix_unitary
    elif name_model == 'FldzhyanBellHalfNxN':
        model = FldzhyanBellHalfNxN(N=N, insersion_loss_MMI=0., imbalance_MMI=0.)
        target_matrix = target_matrix_unitary
    elif name_model == 'NEUROPULSBonus_unitaryNxN':
        model = model = NEUROPULSBonus_unitaryNxN(N=N, insersion_loss_MMI=0., imbalance_MMI=0., insersion_loss_Crossing=0., cross_talk_Crossing=0.)
        target_matrix = target_matrix_unitary
    elif name_model == 'NEUROPULSBonus_unitary_2long_NxN':
        model = model = NEUROPULSBonus_unitary_2long_NxN(N=N, insersion_loss_MMI=0., imbalance_MMI=0., insersion_loss_Crossing=0., cross_talk_Crossing=0.)
        target_matrix = target_matrix_unitary
    # UNIVERSAL MATRIX
    elif name_model == 'NEUROPULSBonus_anymatrixNxN':
        model = model = NEUROPULSBonus_anymatrixNxN(N=N, insersion_loss_MMI=0., imbalance_MMI=0., insersion_loss_Crossing=0., cross_talk_Crossing=0.)
        target_matrix = target_any_matrix
    elif name_model == 'NEUROPULSBonus_Bell_Minht_NxN':
        model = model = NEUROPULSBonus_Bell_Minht_NxN(N=N, insersion_loss_MMI=0., imbalance_MMI=0., insersion_loss_Crossing=0., cross_talk_Crossing=0.)
        target_matrix = target_any_matrix
    elif name_model == 'NEUROPULSBonus_BellNormal_NxN':
        model = model = NEUROPULSBonus_BellNormal_NxN(N=N, insersion_loss_MMI=0., imbalance_MMI=0., insersion_loss_Crossing=0., cross_talk_Crossing=0.)
        target_matrix = target_any_matrix
    elif name_model == 'CB_withMZI':
        model = model = model = Arch_lastMZI_NxN(name_architecture='ClementsBellNxN', N=N, insersion_loss_MMI=0., imbalance_MMI=0., insersion_loss_Crossing=0., cross_talk_Crossing=0.)
        target_matrix = target_any_matrix
    elif name_model == 'FB_withMZI':
        model = model = model = Arch_lastMZI_NxN(name_architecture='FldzhyanBellNxN', N=N, insersion_loss_MMI=0., imbalance_MMI=0., insersion_loss_Crossing=0., cross_talk_Crossing=0.)
        target_matrix = target_any_matrix
    elif name_model == 'FBH_withMZI':
        model = model = model = Arch_lastMZI_NxN(name_architecture='FldzhyanBellHalfNxN', N=N, insersion_loss_MMI=0., imbalance_MMI=0., insersion_loss_Crossing=0., cross_talk_Crossing=0.)
        target_matrix = target_any_matrix
    elif name_model == 'NPBU_withMZI':
        model = model = model = Arch_lastMZI_NxN(name_architecture='NEUROPULSBonus_unitaryNxN', N=N, insersion_loss_MMI=0., imbalance_MMI=0., insersion_loss_Crossing=0., cross_talk_Crossing=0.)
        target_matrix = target_any_matrix
    else:
        raise Exception('Something not good on the input')
    
    loss_fn = CMatrixMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_xepoch_data = []
    for epoch in range(n_epochs):     # Optimiziation with gradient
        prediction_matrix = model()
        loss = loss_fn(prediction_matrix, target_matrix)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_xepoch_data.append(loss.detach().numpy())

    # print(f'Epoch [{epoch+1}/{n_epochs}], i_matrix: {i_matrix}, model: {name_model}, rep: {rep}, Loss: {loss.item():.10f}')

    # Target, predition, loss x epochs
    target_rediction_loss = [('target'+'_name_model'+name_model+'_N'+str(N)+'_i_matrix'+str(i_matrix)+'_rep'+str(rep),
                              target_matrix.numpy())]
    target_rediction_loss.append(('predition'+'_name_model'+name_model+'_N'+str(N)+'_i_matrix'+str(i_matrix)+'_rep'+str(rep),
                                  prediction_matrix.detach().numpy()))
    target_rediction_loss.append(('loss_xepoch'+'_name_model'+name_model+'_N'+str(N)+'_i_matrix'+str(i_matrix)+'_rep'+str(rep),
                                  np.array(loss_xepoch_data)))
    return target_rediction_loss


if __name__ == "__main__":
    # num_processors = multiprocessing.cpu_count()       # Get the number of available processors

    TPL_list = []
    for N in Ns:
        for i_matrix in tqdm.trange(n_matrices):
            # UNITARY MATRIX
            target_matrix_unitary = torch.tensor(unitary_group.rvs(N))    # Unitary matrix
                
            # UNIVERSAL MATRIX
            real_part = torch.randn(N, N)
            imag_part = torch.randn(N, N)
            complex_matrix = torch.complex(real_part, imag_part)
            column_norms = torch.sum((torch.abs(complex_matrix))**2, dim=0)     # Power outputs
            scaling_factor = 0.5 * torch.rand(1) / torch.sqrt(torch.max(column_norms)/2)        # Random scaling between 0.5 and 0!!!!!!!!!!!!!
            target_any_matrix = complex_matrix * scaling_factor
            # print(torch.sum((torch.abs(target_matrix))**2, dim=0))

            # Create the inputs
            args_list = [(name_model, N, i_matrix, rep, target_matrix_unitary, target_any_matrix) for name_model in name_models for rep in range(n_repetitions)]

            # Run all the rep and models in parallel
            target_predition_loss = list(futures.map(model_prediction, args_list))

            TPL_flat = [item for sublist in target_predition_loss for item in sublist]
            # Matrix is [[(T),(P), (L) of mod0-M0-rep0], [(T),(P), (L) of mod0-M0-rep1], etc]
            TPL_list.extend(TPL_flat)

            TPL_np = np.array(TPL_list, dtype=[('label', 'U100'), ('matrix', 'O')])
            np.save(folder_path+'202404116_HPC_simulation', TPL_np)




