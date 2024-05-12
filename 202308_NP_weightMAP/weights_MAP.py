# Derive the limit of the weights in the NEUROPULS architecture

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm         # make loops show as a smart progress meter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)
#torch.cuda.set_device(device)

def increment_base(arr, base):
    carry = 1
    for i in range(len(arr)-1, -1, -1):
        current_sum = arr[i] + carry
        arr[i] = current_sum % base
        carry = current_sum // base
        if carry == 0:
            break
    if carry > 0:
        arr.insert(0, carry)
    return arr


def weight_simulation(model, device, N: int=3, ind_input: int=0, ind_output: int=0):
    # step = 0.4
    # stop = 2*np.pi + step
    # weight_values_array = np.arange(0, stop, step)
    # print(weight_values_array)
    # ind_weight = np.zeros(N*(N-1))
    # print(len(weight_values_array)**len(ind_weight)-1)
    # for i in tqdm.trange(len(weight_values_array)**len(ind_weight)-1):
    #     increment_base(ind_weight, base=len(weight_values_array))
    # print(ind_weight)

    for name, param in model.named_parameters():
        if 'phase_shift' in name:
            # param.data = torch.zeros(N, requires_grad=False, device=device)
            param.data = np.pi*torch.ones(N, requires_grad=False, device=device)

    input_tensor = torch.zeros(N, requires_grad=False, device=device)
    input_tensor[ind_input] = 1.0

    with torch.no_grad():
        output = model(input_tensor)
    output_val = output[ind_output]
    return output_val


def min_gradient(model, device, ind_weigh: list=[0, 0], epochs: int=100, N: int=3, lr: float=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    x = torch.zeros(N, requires_grad=False, device=device)  # Input x_n=1 all other 0
    x[ind_weigh[0]] = 1.0
    min_epoch = np.zeros(epochs)
    loss_epoch = np.zeros(epochs)
    for i in range(epochs):
        output = model(x)
        min_epoch[i] = output[ind_weigh[1]]
        loss = loss_fn(output[ind_weigh[1]], torch.tensor(0.0))
        loss_epoch[i] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return min_epoch, loss_epoch


def max_gradient(model, device, ind_weigh: list=[0, 0], epochs: int=100, N: int=3, lr: float=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    x = torch.zeros(N, requires_grad=False, device=device)  # Input x_n=1 all other 0
    x[ind_weigh[0]] = 1.0
    max_epoch = np.zeros(epochs)
    loss_epoch = np.zeros(epochs)
    for i in range(epochs):
        output = model(x)
        max_epoch[i] = output[ind_weigh[1]]
        loss = loss_fn(output[ind_weigh[1]], torch.tensor(1.0))
        loss_epoch[i] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return max_epoch, loss_epoch

def plot_epoch(value, loss):
    print("Value found:", value[-1])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(value, label="Value")            # Plot Value
    ax1.set_title("Value")
    ax1.set_xlabel("epochs")
    ax1.set_ylim([0.0, 1.05])
    ax1.legend()
    ax2.plot(loss, label="Losses")            # Plot Loss
    ax2.set_title("Loss")
    ax2.set_xlabel("epochs")
    ax2.legend()
    plt.show();

if __name__ == "__main__":
    print("Test")

    # out = weight_simulation(ind_input=0 , ind_output=1)
    # print(out)

    # HYPERPARAMETERS ---------------------------------------------------------------------------------------------
    epochs = 1000
    lr = 0.1
    N = 5

    # from NEUROPULS_2 import NEUROPULSNxN_2_2
    from Clements import ClementsNxN


    max_weight = np.zeros((N,N))
    min_weight = np.ones((N,N))
    for i in tqdm(range(N*N)):
        max_losse = max_gradient(ClementsNxN(N), device=device, ind_weigh=[i//N, i%N], epochs=epochs, N=N, lr=lr)
        # min_losse = min_gradient(NEUROPULSNxN_2_2(N), ind_weigh=[i//N, i%N], epochs=epochs, N=N, lr=lr)
        max_weight[i//N, i%N] = max_losse[0][-1]
        # min_weight[i//N, i%N] = min_losse[0][-1]

    
    print(max_weight)
    # print(min_weight)
    
    # -------------------------------------------------------------------------------------------------------------

