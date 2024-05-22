# Tracking the random target matrix
# Basically copy this code to all the CPU and the different CPU
# I know whick CPU is working base on the CPU rank number 0-n
# to run the code: mpirun -n 12 python ./<your_code>.py
# run this the task.jason where it will run the the .sh script: Ctrl+Shift+B
# to run directly terminal comand (after make it executable chmod +x your_script.sh): bash your_script.sh
# To run on the HPC TERMINAL TO PUT QUEUE RUN FUCK: qsub your_script.sh

import os
import json
import sys
from mpi4py import MPI
import numpy as np
from scipy.stats import unitary_group
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from unitary_matrix_models import *


# MPI inizialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()      # Get the rank of the current process within the communicator
size_comm = comm.Get_size()      # Get the total number of processes in the communicator

# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load some hyperparameters ---------------------------------------------------------------------------------------
def load_hyperparameters(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)


# Loss function ---------------------------------------------------------------------------------------------------
# Compless MSE Loss function, from the definition
class CMatrixMSELoss(nn.Module):
    def __init__(self):
        super(CMatrixMSELoss, self).__init__()
    
    def forward(self, predicted_matrix, target_matrix):
        mag_diff_sq = torch.abs(predicted_matrix - target_matrix)**2
        loss = torch.sum(mag_diff_sq) / torch.numel(target_matrix)
        return loss


# Model -----------------------------------------------------------------------------------------------------------
def select_model(name_model):
    mmi_i_losses_mtx_even = torch.full((n_inputs, n_inputs//2), i_loss)
    mmi_i_losses_mtx_odd = torch.full((n_inputs, n_inputs//2-1), i_loss)
    mmi_imbalances_mtx_even = torch.full((n_inputs, n_inputs//2), imbalance)
    mmi_imbalances_mtx_odd = torch.full((n_inputs, n_inputs//2-1), imbalance)
    crossing_i_losses_mtx_odd = torch.full((n_inputs//2-1, n_inputs//2+1), i_loss)
    crossing_crosstalks_mtx_odd = torch.full((n_inputs//2-1, n_inputs//2+1), cross_talk)
    if name_model == 'Clements_Arct':
        model = Clements_Arct(
            n_inputs=n_inputs,
            mmi_i_losses_mtx_even=mmi_i_losses_mtx_even,
            mmi_i_losses_mtx_odd=mmi_i_losses_mtx_odd,
            mmi_imbalances_mtx_even=mmi_imbalances_mtx_even,
            mmi_imbalances_mtx_odd=mmi_imbalances_mtx_odd)
    elif name_model == 'ClementsBell_Arct':
        model = ClementsBell_Arct(
            n_inputs=n_inputs,
            mmi_i_losses_mtx_even=mmi_i_losses_mtx_even,
            mmi_i_losses_mtx_odd=mmi_i_losses_mtx_odd,
            mmi_imbalances_mtx_even=mmi_imbalances_mtx_even,
            mmi_imbalances_mtx_odd=mmi_imbalances_mtx_odd)
    elif name_model == 'Fldzhyan_Arct':
        model = Fldzhyan_Arct(
            n_inputs=n_inputs,
            mmi_i_losses_mtx_even=mmi_i_losses_mtx_even,
            mmi_i_losses_mtx_odd=mmi_i_losses_mtx_odd,
            mmi_imbalances_mtx_even=mmi_imbalances_mtx_even,
            mmi_imbalances_mtx_odd=mmi_imbalances_mtx_odd)
    elif name_model == 'FldzhyanBell_Arct':
        model = FldzhyanBell_Arct(
            n_inputs=n_inputs,
            mmi_i_losses_mtx_even=mmi_i_losses_mtx_even,
            mmi_i_losses_mtx_odd=mmi_i_losses_mtx_odd,
            mmi_imbalances_mtx_even=mmi_imbalances_mtx_even,
            mmi_imbalances_mtx_odd=mmi_imbalances_mtx_odd)
    elif name_model == 'FldzhyanBellHalf_Arct':
        model = FldzhyanBellHalf_Arct(
            n_inputs=n_inputs,
            mmi_i_losses_mtx_even=mmi_i_losses_mtx_even,
            mmi_i_losses_mtx_odd=mmi_i_losses_mtx_odd,
            mmi_imbalances_mtx_even=mmi_imbalances_mtx_even,
            mmi_imbalances_mtx_odd=mmi_imbalances_mtx_odd)
    elif name_model == 'NEUROPULS_Arct':
        model = NEUROPULS_Arct(
            n_inputs=n_inputs,
            mmi_i_losses_mtx_even=mmi_i_losses_mtx_even,
            mmi_imbalances_mtx_even=mmi_imbalances_mtx_even,
            crossing_i_losses_mtx_odd=crossing_i_losses_mtx_odd,
            crossing_crosstalks_mtx_odd=crossing_crosstalks_mtx_odd)
    elif name_model == 'NEUROPULSBell_Arct':
        model = NEUROPULSBell_Arct(
            n_inputs=n_inputs,
            mmi_i_losses_mtx_even=mmi_i_losses_mtx_even,
            mmi_imbalances_mtx_even=mmi_imbalances_mtx_even,
            crossing_i_losses_mtx_odd=crossing_i_losses_mtx_odd,
            crossing_crosstalks_mtx_odd=crossing_crosstalks_mtx_odd)
    else:
        model = None
        raise Exception('Something not good on the input')
    return model


# Calculate the prediction ----------------------------------------------------------------------------------------
def model_prediction(args):
    name_model, n_inputs, i_matrix, rep, target_matrix = args
    # Inizialize model
    model = select_model(name_model=name_model)
    # Create loss and optimizer
    loss_fn = CMatrixMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Optimization
    for _ in range(n_epochs):     # Optimiziation with gradient
        prediction_matrix = model()
        loss = loss_fn(prediction_matrix, target_matrix)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Save data
    target_pediction = [
        ('target'+'_name_model'+name_model+'_nin'+str(n_inputs)+'_imatrix'+str(i_matrix)+'_rep'+str(rep),
         target_matrix.numpy()),
        ('prediction'+'_name_model'+name_model+'_nin'+str(n_inputs)+'_imatrix'+str(i_matrix)+'_rep'+str(rep),
         prediction_matrix.detach().numpy())]
    return target_pediction


# Create all target -----------------------------------------------------------------------------------------------
# Function create all the targets input and the corrispective input_list
def input_all_target_matrix():
    input_list = []
    for i_matrix in range(n_matrices):
        # Haar-random unitary matrices of any size
        target_matrix_unitary = torch.tensor(unitary_group.rvs(n_inputs))
        input_list.extend([(name_model, n_inputs, i_matrix, rep, target_matrix_unitary)
                            for name_model in name_models
                            for rep in range(n_repetitions)])
    return input_list


# Split function --------------------------------------------------------------------------------------------------
# Function to split the input. Ex out: [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]] first has always bigger dim
def split_into_sublists(data, n):
    k, m = divmod(len(data), n)
    return [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


# Test ------------------------------------------------------------------------------------------------------------
# To test the saving process
def test_out_TP():
    out_TP = [
        ("target", np.zeros((n_inputs, n_inputs), dtype='complex')),
        ("predition", np.zeros((n_inputs, n_inputs), dtype='complex')),]
    return out_TP


# =================================================================================================================
# =================================================== Main ========================================================
# =================================================================================================================
if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python main_program.py <config_file>")
        sys.exit(1)
    config_file = sys.argv[1]

    # HIGH PARAMETERS =============================================================================================
    hp = load_hyperparameters(config_file)
    seed = 37

    n_inputs = hp['n_inputs']
    n_matrices = 1000
    n_repetitions = 5

    lr = 0.001
    if n_inputs == 4:
        n_epochs = 20000
    elif n_inputs == 8:
        n_epochs = 22000
    elif n_inputs == 16:
        n_epochs = 25000

    name_models = ['Clements_Arct', 'ClementsBell_Arct', 'Fldzhyan_Arct', 'FldzhyanBell_Arct',
                   'FldzhyanBellHalf_Arct', 'NEUROPULS_Arct', 'NEUROPULSBell_Arct']

    # CONSTANT LOSS
    i_loss = hp['i_loss']          # from 0 min to 1 max
    imbalance = hp['imbalance']       # from -0.5 min to 0.5 max
    cross_talk = imbalance      # from 0 min to 1 max

    # If RAM too full decrease and fail with kill problem -> Decrease this number
    n_bachup = 500

    folder_path = "outdata/"
    name_folder_out = 'n'+str(n_inputs)+'_iloss'+str(i_loss)+'_imb'+str(imbalance)+'_HPC_simulation/'
    # =============================================================================================================
    # =============================================================================================================


    # Seed for the random number
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_sublist = None    # Inizialize between all the CPUs
    maxn_sim_core = None   # The maximum sim alway on CPU 1
    # Just CPU 0 will create the input 
    if rank == 0:
        # Directory to save all data
        if not os.path.exists(folder_path+name_folder_out):
            os.makedirs(folder_path+name_folder_out)
        input_list = input_all_target_matrix()
        input_sublist = split_into_sublists(input_list, size_comm)
        del input_list
        maxn_sim_core = len(input_sublist[0])
    
    # Take the list compose by 12 sublist and spread them between the 12 CPUs (12 is jsut example. It is size_comm)
    input_list = comm.scatter(input_sublist, root=0)     # Scatter data from rank 0 to all other ranks
    maxn_sim_core = comm.scatter([maxn_sim_core for i in range(size_comm)], root=0)   # Max num simulation


    # ============================================== SIMULATIONS ==================================================
    # Two case CORE < SIM nMAX_sim_xCORE=1 and CORE > SIM nMAX_sim_xCORE LAS = +n sim to do
    out_targets_predictions = []
    if rank == 0:   # Just made to have the line to see number simulation and extimation timing
        for i in tqdm.trange(maxn_sim_core):
            if i < len(input_list):
                args = input_list[i]
                out_targets_predictions.extend(model_prediction(args=args))
            # Save intermidiate result
            if i%n_bachup == 0 and i != 0:
                targets_predictions_np = np.array(out_targets_predictions, dtype=[('label', 'U100'), ('matrix', 'O')])
                out_targets_predictions = []
                np.save(folder_path+name_folder_out+'save'+str(i//n_bachup)+'_CPU'+str(rank), targets_predictions_np)
                del targets_predictions_np

    else:
        for i in range(maxn_sim_core):
            if i < len(input_list):
                args = input_list[i]
                out_targets_predictions.extend(model_prediction(args=args))
            # Save intermidiate result
            if i%n_bachup == 0 and i != 0:
                targets_predictions_np = np.array(out_targets_predictions, dtype=[('label', 'U100'), ('matrix', 'O')])
                out_targets_predictions = []
                np.save(folder_path+name_folder_out+'save'+str(i//n_bachup)+'_CPU'+str(rank), targets_predictions_np)
                del targets_predictions_np

    if out_targets_predictions != []:
        targets_predictions_np = np.array(out_targets_predictions, dtype=[('label', 'U100'), ('matrix', 'O')])
        out_targets_predictions = []
        np.save(folder_path+name_folder_out+'save'+str(i//n_bachup)+'_CPU'+str(rank), targets_predictions_np)
        del targets_predictions_np
    
    if rank == 0:
        print("!!!Congratulation it has FINISH correctly!!!")



