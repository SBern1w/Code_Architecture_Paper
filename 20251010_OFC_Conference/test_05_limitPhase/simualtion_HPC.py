# =================================================================================================================
# Goal: Limit the phase between 0 and pi
# Check:

# TODO:

# =================================================================================================================

import os
import argparse
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import time
from datetime import timedelta
import matplotlib.pyplot as plt

from models.unitary_matrix_models import *


# Number of simulation run
parser = argparse.ArgumentParser(description='Process the hyperparameters.')
parser.add_argument('--run_index', required=True, help='run_index')
args = parser.parse_args()
run_index = int(args.run_index)

start_time = time.time()

# =================================================================================================================
# =========================================== HYPARAMETERS ========================================================
# =================================================================================================================
# Each CPU run 10 repetitions and 10 different matrices
n_CPU_X_sim = 100
n_matrix_x_CPU = 10
n_repetitions = 10

configs = [
    {"index": 0, "model_obj": Clements_Arct, "type_lim": "clamp_ste"},
    {"index": 1, "model_obj": Clements_Arct, "type_lim": "clamp"},
    {"index": 2, "model_obj": Clements_Arct, "type_lim": "tanh_soft"},
    {"index": 3, "model_obj": Clements_Arct, "type_lim": "tanh_hard"},
    
    {"index": 4, "model_obj": Fldzhyan_Arct, "type_lim": "clamp_ste"},
    {"index": 5, "model_obj": Fldzhyan_Arct, "type_lim": "clamp"},
    {"index": 6, "model_obj": Fldzhyan_Arct, "type_lim": "tanh_soft"},
    {"index": 7, "model_obj": Fldzhyan_Arct, "type_lim": "tanh_hard"},
    
    {"index": 8, "model_obj": NEUROPULSCrossingSide_Arct, "type_lim": "clamp_ste"},
    {"index": 9, "model_obj": NEUROPULSCrossingSide_Arct, "type_lim": "clamp"},
    {"index": 10, "model_obj": NEUROPULSCrossingSide_Arct, "type_lim": "tanh_soft"},
    {"index": 11, "model_obj": NEUROPULSCrossingSide_Arct, "type_lim": "tanh_hard"},
]

# Each CPU
search_index = run_index // n_CPU_X_sim
config = next(c for c in configs if c["index"] == search_index)
model_obj = config["model_obj"]
num_folder = config["index"]
type_lim = config["type_lim"]

# =================================================================================================================
n_inputs = 8

lr = 0.001
n_epochs = 500 * n_inputs + 18000

# GAUSSIAN DISTRIBUTION
pc_iloss_mu = 0.
pc_iloss_sigma = 0.
i_loss_MMI_mu = 0.
i_loss_MMI_sigma = 0.
imbalance_mu = 0.
imbalance_sigma = 0.
i_loss_Crossing_mu = 0.
i_loss_Crossing_sigma = 0.
cross_talk_mu = -1000.
cross_talk_sigma = 0.

# pc_iloss_mu = 0.        # Average =P_out/P_in. 0dB pefect component, -100dB very lossy
# pc_iloss_sigma = 0.     # Std deviation

# i_loss_MMI_mu = -0.5          # Average =P_out/P_in. 0dB pefect component, -100dB very lossy
# i_loss_MMI_sigma = 0.1       # Std deviation
# imbalance_mu = 0.           # Average =P_outmax/P_outmin. 0dB 50/50 MMI, 100dB all power to outUP, -100dB all power to outDOWN
# imbalance_sigma = 0.15        # Std deviation

# i_loss_Crossing_mu = -0.2         # Average =P_out/P_in. 0dB pefect component, -100dB very lossy
# i_loss_Crossing_sigma = 0.05      # Std deviation
# cross_talk_mu = -35.          # Average =P_leakout/P_otherout. -infdB Crossing perfect, -1dB very bad device a lot power leak
# cross_talk_sigma = 1.           # Std deviation

# =================================================================================================================
# =================================================================================================================
# =================================================================================================================

# Load the right target -------------------------------------------------------------------------------------------
def load_targets(index_matrix, n_inputs):
    all_target = torch.load("./dataset/targets_nIN"+str(n_inputs)+"_nM1000.pt", weights_only=False)
    target_matricies = all_target[index_matrix*n_matrix_x_CPU:(index_matrix+1)*n_matrix_x_CPU, : , :]
    return target_matricies

# Select model ----------------------------------------------------------------------------------------------------
# Truncate the gaussian distribution
def create_truncated_gaussian_tensor(mu, sigma, shape, max_value=None):
    tensor = torch.normal(mean=mu, std=sigma, size=shape)
    if max_value is not None:
        tensor = torch.clamp(tensor, max=max_value)
    return tensor

def select_model(name_model):
    mmi_i_losses_mtx_even = create_truncated_gaussian_tensor(i_loss_MMI_mu, i_loss_MMI_sigma, (2*(n_inputs-1), n_inputs//2), 0)
    mmi_i_losses_mtx_even = 10**(mmi_i_losses_mtx_even/10)
    mmi_i_losses_mtx_odd = create_truncated_gaussian_tensor(i_loss_MMI_mu, i_loss_MMI_sigma, (n_inputs, n_inputs//2-1), 0)
    mmi_i_losses_mtx_odd = 10**(mmi_i_losses_mtx_odd/10)
    mmi_imbalances_mtx_even = create_truncated_gaussian_tensor(imbalance_mu, imbalance_sigma, (2*(n_inputs-1), n_inputs//2))
    mmi_imbalances_mtx_even = 10**(mmi_imbalances_mtx_even/10)
    mmi_imbalances_mtx_odd = create_truncated_gaussian_tensor(imbalance_mu, imbalance_sigma, (n_inputs, n_inputs//2-1))
    mmi_imbalances_mtx_odd = 10**(mmi_imbalances_mtx_odd/10)

    crossing_i_losses_mtx_odd = create_truncated_gaussian_tensor(i_loss_Crossing_mu, i_loss_Crossing_sigma, (n_inputs-2, n_inputs//2-1), 0)
    crossing_i_losses_mtx_odd = 10**(crossing_i_losses_mtx_odd/10)
    crossing_i_losses_mtx_odd_side = create_truncated_gaussian_tensor(i_loss_Crossing_mu, i_loss_Crossing_sigma, (n_inputs-2, n_inputs//2+1), 0)
    crossing_i_losses_mtx_odd_side = 10**(crossing_i_losses_mtx_odd_side/10)
    crossing_crosstalks_mtx_odd = create_truncated_gaussian_tensor(cross_talk_mu, cross_talk_sigma, (n_inputs-2, n_inputs//2-1))
    crossing_crosstalks_mtx_odd = 10**(crossing_crosstalks_mtx_odd/10)
    crossing_crosstalks_mtx_odd_side = create_truncated_gaussian_tensor(cross_talk_mu, cross_talk_sigma, (n_inputs-2, n_inputs//2+1))
    crossing_crosstalks_mtx_odd_side = 10**(crossing_crosstalks_mtx_odd_side/10)

    if name_model == Clements_Arct:
        model = Clements_Arct(
            n_inputs=n_inputs,
            mmi_i_losses_mtx_even=mmi_i_losses_mtx_even,
            mmi_i_losses_mtx_odd=mmi_i_losses_mtx_odd,
            mmi_imbalances_mtx_even=mmi_imbalances_mtx_even,
            mmi_imbalances_mtx_odd=mmi_imbalances_mtx_odd,
            type_lim=type_lim)
    elif name_model == Fldzhyan_Arct:
        model = Fldzhyan_Arct(
            n_inputs=n_inputs,
            mmi_i_losses_mtx_even=mmi_i_losses_mtx_even,
            mmi_i_losses_mtx_odd=mmi_i_losses_mtx_odd,
            mmi_imbalances_mtx_even=mmi_imbalances_mtx_even,
            mmi_imbalances_mtx_odd=mmi_imbalances_mtx_odd,
            type_lim=type_lim)
    elif name_model == NEUROPULSCrossingSide_Arct:
        model = NEUROPULSCrossingSide_Arct(
            n_inputs=n_inputs,
            mmi_i_losses_mtx_even=mmi_i_losses_mtx_even,
            mmi_imbalances_mtx_even=mmi_imbalances_mtx_even,
            crossing_i_losses_mtx_odd=crossing_i_losses_mtx_odd_side,
            crossing_crosstalks_mtx_odd=crossing_crosstalks_mtx_odd_side,
            type_lim=type_lim)
    else:
        model = None
        raise Exception('Something not good on the input')
    return model

# Fidelity and Loss function --------------------------------------------------------------------------------------
def FidelityUnitary(predicted_matrix, target_matrix):
    n_inputs = predicted_matrix.shape[0]
    predicted_matrix = predicted_matrix.to(torch.complex128)
    target_matrix = target_matrix.to(torch.complex128)
    Frobenius_module_p = torch.trace(torch.matmul(predicted_matrix.t().conj(), predicted_matrix))
    Frobenius_pt = torch.trace(torch.matmul(predicted_matrix.t().conj(), target_matrix))
    cosine_similarity = (torch.abs(Frobenius_pt))**2/(n_inputs*Frobenius_module_p)
    Fidelity = torch.abs(cosine_similarity)
    return Fidelity

def Loss_FildelityUnitary(predicted_matrix, target_matrix):
    fidelity = FidelityUnitary(predicted_matrix, target_matrix)
    return 1 - fidelity

# Calculate the prediction ----------------------------------------------------------------------------------------
def model_training(model, optimizer, target_matrix):
    for _ in range(n_epochs):     # Optimiziation with gradient
        pred_matrix = model()
        loss = Loss_FildelityUnitary(pred_matrix, target_matrix)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Get fidelity
    with torch.no_grad():
        pred_matrix = model()
        fidelity = FidelityUnitary(pred_matrix, target_matrix)
    return fidelity, model


# =================================================================================================================
# =============================================== MAIN ============================================================
# =================================================================================================================
if __name__ == "__main__":
    # Get the right data
    index_matrix = run_index % n_CPU_X_sim
    targets = load_targets(index_matrix, n_inputs)

    # Simulations
    fidelities = []
    n_targets = targets.shape[0]
    for idx_targ in tqdm.trange(n_targets):
        for rep in range(n_repetitions):
            # Initialize new model with different initial phase shifts for each simulaiton
            model = select_model(model_obj)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            last_fidelity, model = model_training(model, optimizer, targets[idx_targ, : , :])
            fidelities.append(last_fidelity)

            # chagne the phase shift gradually to see effect on fidelity
    
    # Save the results model ======================================================================================
    # Create folder and retun the directory:
    base_dir = "./outdata/"
    # Create the new run directory
    run_dir = os.path.join(base_dir, f'run{num_folder}')
    os.makedirs(run_dir, exist_ok=True)
    # Create the full path
    filename = f'simulation_{run_index}.pt'
    full_path = os.path.join(run_dir, filename)
    # Create a dictionary
    save_dict = {
        'model_name': model_obj.__name__,
        'run_index': run_index,
        'fidelities': fidelities,
    }
    # Save it
    torch.save(save_dict, full_path)

    end_time = time.time()
    work_duration = end_time - start_time
    max_duration_human_readable = str(timedelta(seconds=work_duration))
    print(f"The maximum work duration is {max_duration_human_readable} (HH:MM:SS).")
    
    print("Yeeeeh the code has finished!!!")

