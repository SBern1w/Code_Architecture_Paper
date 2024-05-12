# Tracking the random target matrix
# Basically copy this code to all the CPU and the different CPU
# I know whick CPU is working base on the CPU rank number 0-n
# to run the code: mpirun -n 12 python ./<your_code>.py
# run this the task.jason where it will run the the .sh script: Ctrl+Shift+B
# to run directly terminal comand (after make it executable chmod +x your_script.sh): bash your_script.sh
# To run on the TERMINAL TO PUT QUEUE RUN FUCK: qsub your_script.sh

from mpi4py import MPI
import numpy as np
from scipy.stats import unitary_group
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from matrix_models import *


# HIGH PARAMETERS =================================================================================================
seed = 37

# Hyparameters of the losses
lr = 0.001
n_epochs = 20000

name_models = ['ClementsBellNxN', 'FldzhyanBellNxN', 'FldzhyanBellHalfNxN', 'NEUROPULSBonus_unitaryNxN', 'NEUROPULSBonus_unitary_2long_NxN',
               'NEUROPULSBonus_anymatrixNxN', 'NEUROPULSBonus_Bell_Minht_NxN', 'NEUROPULSBonus_BellNormal_NxN',
               'CB_withMZI', 'FB_withMZI', 'FBH_withMZI', 'NPBU_withMZI',]
# name_models = ['ClementsBellNxN', 'FldzhyanBellNxN', 'FldzhyanBellHalfNxN', 'NEUROPULSBonus_unitaryNxN', 'NEUROPULSBonus_unitary_2long_NxN',
#                'NEUROPULSBonus_anymatrixNxN', 'NEUROPULSBonus_Bell_Minht_NxN', 'NEUROPULSBonus_BellNormal_NxN', 'CB_withMZI']

N = 4
n_matrices = 1000
n_repetitions = 10

# folder_path = '20240403_matrix_optimization/outdata/20240409_data/'
folder_path = 'outdata/'
name_file_out = '202404117_HPC_simulation'
# =================================================================================================================



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


# Function create all the targets input and the corrispective input_list
def input_all_target_matrix():
    input_list = []
    for i_matrix in range(n_matrices):
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

        input_list.extend([(name_model, N, i_matrix, rep, target_matrix_unitary, target_any_matrix) for name_model in name_models for rep in range(n_repetitions)])
    return input_list


# =================================================== Main ========================================================
if __name__ == "__main__":
    comm = MPI.COMM_WORLD       # Initialize the MPI communicator for all processes
    rank = comm.Get_rank()      # Get the rank of the current process within the communicator
    size_comm = comm.Get_size()      # Get the total number of processes in the communicator

    input_sublist = None    # Inizialize between all the CPUs
    nMAX_sim_xCORE = 0
    if rank==0:     # Just CPU 0 will create the same matrix
        torch.manual_seed(seed)      # Seed for the random number
        np.random.seed(seed)
        input_list = input_all_target_matrix()
        input_sublist = [input_list[i * len(input_list)//size_comm: (i + 1) * len(input_list) // size_comm] for i in range(size_comm)]
        nMAX_sim_xCORE = len(input_sublist[0]) if len(input_sublist[0]) > len(input_sublist[size_comm-1]) else len(input_sublist[size_comm-1])

    else:           #, if I don't make this all the repetions will start with th same point
        torch.manual_seed(seed+rank)
        np.random.seed(seed+rank)
    
    # Take the list compose by 12 sublist and spread them between the 12 CPUs (12 is jsut example. It is size_comm)
    input_list = comm.scatter(input_sublist, root=0)     # Scatter data from rank 0 to all other ranks
    nMAX_sim_xCORE = comm.scatter([nMAX_sim_xCORE for i in range(size_comm)], root=0)   # Max num simulation
  
    # ============================================== SIMULATIONS ==================================================
    n_bachup = 200
    # Two case CORE < SIM nMAX_sim_xCORE=1 and CORE > SIM nMAX_sim_xCORE LAS = +n sim to do
    out_TPL = None
    if rank == 0:   # Just made to have the line to see number simulation and extimation timing
        for i in tqdm.trange(nMAX_sim_xCORE):
            try:
                args = input_list[i]
                out_TPL = model_prediction(args=args)
            except IndexError:
                out_TPL = []

            if i%n_bachup == 0:      # To save intermidiate results
                out_TPL_gathered = comm.gather(out_TPL, root=0)

                # Save intermidiate result
                TPL_np = np.array(out_TPL_gathered, dtype=[('label', 'U100'), ('matrix', 'O')])
                np.save(folder_path+name_file_out, TPL_np)
    else:
        for i in range(nMAX_sim_xCORE):
            try:
                args = input_list[i]
                out_TPL = model_prediction(args=args)
            except IndexError:
                out_TPL = []
            
            if i%n_bachup == 0:      # To save intermidiate results
                comm.gather(out_TPL, root=0)
    
    # Final
    out_TPL_gathered = comm.gather(out_TPL, root=0)
    TPL_np = np.array(out_TPL_gathered, dtype=[('label', 'U100'), ('matrix', 'O')])
    np.save(folder_path+name_file_out, TPL_np)

