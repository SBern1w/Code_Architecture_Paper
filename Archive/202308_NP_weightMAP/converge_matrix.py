# Converge the model to the the target matrix


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import unitary_group

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm         # make loops show as a smart progress meter

from architecture_matrix import *




def complex_mse_loss(predictions, targets):
    # Compute the squared absolute differences element-wise
    squared_abs_diffs = torch.abs(predictions - targets)**2

    # Compute the mean of the squared absolute differences
    mse_loss = torch.mean(squared_abs_diffs)

    return mse_loss

def complex_and_real_loss(predictions, targets):
    # Compute the element-wise absolute differences of real and imaginary parts
    real_diffs = torch.abs(predictions.real - targets.real)
    imag_diffs = torch.abs(predictions.imag - targets.imag)

    # Compute the Frobenius norm of the absolute differences
    frobenius_loss = torch.norm(real_diffs, 'fro') + torch.norm(imag_diffs, 'fro')

    return frobenius_loss

def complex_frobenius_loss(predictions, targets):
    # Compute the element-wise absolute differences
    abs_diffs = torch.abs(predictions - targets)

    # Compute the Frobenius norm of the absolute differences
    frobenius_loss = torch.norm(abs_diffs, 'fro')

    return frobenius_loss

class ComplexMatrixMSELoss(nn.Module):
    def __init__(self):
        super(ComplexMatrixMSELoss, self).__init__()

    def forward(self, predicted_matrix, target_matrix):
        real_loss = torch.mean((predicted_matrix.real - target_matrix.real)**2)
        imag_loss = torch.mean((predicted_matrix.imag - target_matrix.imag)**2)
        loss = real_loss + imag_loss
        return loss

class ComplexHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(ComplexHuberLoss, self).__init__()
        self.delta = delta

    def forward(self, predicted_matrix, target_matrix):
        diff_real = predicted_matrix.real - target_matrix.real
        diff_imag = predicted_matrix.imag - target_matrix.imag
        diff_magnitude = torch.sqrt(diff_real ** 2 + diff_imag ** 2)
        
        mask = (diff_magnitude < self.delta).float()
        loss = mask * 0.5 * diff_magnitude ** 2 + (1 - mask) * (self.delta * diff_magnitude - 0.5 * self.delta ** 2)
        
        return torch.mean(loss)


N=6
# model = NEUROPULSNxN_2(N=N, MMI_i_loss_dB=0., MMI_imbal_dB=0.)
# model = ClementsNxN(N=N, MMI_i_loss_dB=0., MMI_imbal_dB=0.)
# model = BellNxN(N=N, MMI_i_loss_dB=0., MMI_imbal_dB=0.)
model = BonusIdeaNxN(N=N, MMI_i_loss_dB=0., MMI_imbal_dB=0.)
# model = BonusIdea_2_NxN(N=N, MMI_i_loss_dB=0., MMI_imbal_dB=0.)


# loss_fn = nn.MSELoss()
# loss_fn = ComplexMatrixMSELoss()
loss_fn = ComplexHuberLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)


# target = torch.tensor(unitary_group.rvs(N))     # Unitary matrix
real_part = torch.rand(N, N) * 2 - 1
imaginary_part = torch.rand(N, N) * 2 - 1
complex_matrix = torch.complex(real_part, imaginary_part)
row_sums = torch.sum(torch.abs(complex_matrix), dim=1, keepdim=True)
maximum_value  = 7
random_value = max(row_sums) + (maximum_value  - max(row_sums)) * torch.rand(1)
complex_matrix = complex_matrix / random_value
target = complex_matrix      # Universal matrix
# print(torch.abs(complex_matrix))


# targ = target.detach().numpy()
# print(np.dot(targ, targ.conj().T))

n_epochs = 10000
for epoch in tqdm.trange(n_epochs):
    matrix_pred = model()
    loss = loss_fn(matrix_pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # if (epoch + 1) % 100 == 0:
    #     print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')


    
print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
for idx_m, col_matrix in enumerate(matrix_pred):
    print(f'Target      : {target[idx_m].detach().numpy()}')
    print(f'Prediction  : {matrix_pred[idx_m].detach().numpy()}\n')








