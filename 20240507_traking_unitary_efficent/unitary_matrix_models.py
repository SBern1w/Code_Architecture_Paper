import numpy as np
import torch
import torch.nn as nn

from photonic_components.PhaseChangers_layers import *
from photonic_components.MMI_layers import *
from photonic_components.Crossing_layers import *

# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Clements Architecture -------------------------------------------------------------------------------------------
class Clements_Arct(nn.Module):
    r"""
    Clements architecture:
        0__[]__  __[]__  __________________[]__  __[]__  __________________[]__0
               \/      \/                      \/      \/
        1______/\______/\__[]__  __[]__  ______/\______/\__[]__  __[]__  __[]__1
                               \/      \/                      \/      \/
        2__[]__  __[]__  ______/\______/\__[]__  __[]__  ______/\______/\__[]__2
               \/      \/                      \/      \/
        3______/\______/\______________________/\______/\__________________[]__3

        with:
            0__[]__1 = Phase Shifter

            0__  __0
               \/    =  MMI
            1__/\__1
    """
    def __init__(self,
                 n_inputs: int,
                 mmi_i_losses_mtx_even: torch.Tensor = None,
                 mmi_i_losses_mtx_odd: torch.Tensor = None,
                 mmi_imbalances_mtx_even: torch.Tensor = None,
                 mmi_imbalances_mtx_odd: torch.Tensor = None):
        super(Clements_Arct, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        self._n_inputs = n_inputs
        self._n_layers = n_inputs
        
        # EVEN
        self._ht_layer_even = nn.ModuleList([HeaterLayerMatrix_Even(n_inputs=n_inputs) for _ in range(self._n_layers)])
        if (mmi_i_losses_mtx_even is not None) and (mmi_imbalances_mtx_even is not None):
            self._mmi_layer_even = nn.ModuleList([MMILayerMatrix_Even(
                n_inputs=n_inputs,
                mmi_i_losses=mmi_i_losses_mtx_even[i],
                mmi_imbalances=mmi_imbalances_mtx_even[i])
                for i in range(self._n_layers)])
        else:
            self._mmi_layer_even = nn.ModuleList([MMILayerMatrix_Even(n_inputs=n_inputs) for _ in range(self._n_layers)])
        # ODD
        self._ht_layer_odd = nn.ModuleList([HeaterLayerMatrix_Odd(n_inputs=n_inputs) for _ in range(self._n_layers)])
        if (mmi_i_losses_mtx_odd is not None) and (mmi_imbalances_mtx_odd is not None):
            self._mmi_layer_odd = nn.ModuleList([MMILayerMatrix_Odd(
                n_inputs=n_inputs,
                mmi_i_losses=mmi_i_losses_mtx_odd[i],
                mmi_imbalances=mmi_imbalances_mtx_odd[i])
                for i in range(self._n_layers)])
        else:
            self._mmi_layer_odd = nn.ModuleList([MMILayerMatrix_Odd(n_inputs=n_inputs) for _ in range(self._n_layers)])
        # OUT
        self._ht_layer_out = HeaterLayerMatrix_Full(n_inputs=n_inputs)

    def forward(self):
        id_ini = torch.eye(self._n_inputs, device=device)      # Identity matrix
        arct_matrix = torch.complex(id_ini, torch.zeros_like(id_ini, device=device))
        for i in range(self._n_layers//2):
            arct_matrix = self._ht_layer_even[2*i](arct_matrix)
            arct_matrix = self._mmi_layer_even[2*i](arct_matrix)
            arct_matrix = self._ht_layer_even[2*i+1](arct_matrix)
            arct_matrix = self._mmi_layer_even[2*i+1](arct_matrix)
            arct_matrix = self._ht_layer_odd[2*i](arct_matrix)
            arct_matrix = self._mmi_layer_odd[2*i](arct_matrix)
            arct_matrix = self._ht_layer_odd[2*i+1](arct_matrix)
            arct_matrix = self._mmi_layer_odd[2*i+1](arct_matrix)
        arct_matrix = self._ht_layer_out(arct_matrix)
        return arct_matrix



# Clements Bell Architecture --------------------------------------------------------------------------------------
class ClementsBell_Arct(nn.Module):
    r"""
    Clements Bell architecture:
        0__[]__  __[]__  __________[]__________  __[]__  __________[]______[]__0
               \/      \/                      \/      \/
        1__[]__/\__[]__/\______  __[]__  ______/\__[]__/\______  __[]__  __[]__1
                               \/      \/                      \/      \/
        2__[]__  __[]__  ______/\__[]__/\______  __[]__  ______/\__[]__/\__[]__2
               \/      \/                      \/      \/
        3__[]__/\__[]__/\__________[]__________/\__[]__/\__________[]______[]__3

        with:
            0__[]__1 = Phase Shifter

            0__  __0
               \/    =  MMI
            1__/\__1
    """
    def __init__(self,
                 n_inputs: int,
                 mmi_i_losses_mtx_even: torch.Tensor = None,
                 mmi_i_losses_mtx_odd: torch.Tensor = None,
                 mmi_imbalances_mtx_even: torch.Tensor = None,
                 mmi_imbalances_mtx_odd: torch.Tensor = None):
        super(ClementsBell_Arct, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        self._n_inputs = n_inputs
        self._n_layers = n_inputs
        
        self._ht_layer_full = nn.ModuleList([HeaterLayerMatrix_Full(n_inputs=n_inputs) for _ in range(self._n_layers)])
        # EVEN
        if (mmi_i_losses_mtx_even is not None) and (mmi_imbalances_mtx_even is not None):
            self._mmi_layer_even = nn.ModuleList([MMILayerMatrix_Even(
                n_inputs=n_inputs,
                mmi_i_losses=mmi_i_losses_mtx_even[i],
                mmi_imbalances=mmi_imbalances_mtx_even[i])
                for i in range(self._n_layers)])
        else:
            self._mmi_layer_even = nn.ModuleList([MMILayerMatrix_Even(n_inputs=n_inputs) for _ in range(self._n_layers)])
        # ODD
        if (mmi_i_losses_mtx_odd is not None) and (mmi_imbalances_mtx_odd is not None):
            self._mmi_layer_odd = nn.ModuleList([MMILayerMatrix_Odd(
                n_inputs=n_inputs,
                mmi_i_losses=mmi_i_losses_mtx_odd[i],
                mmi_imbalances=mmi_imbalances_mtx_odd[i])
                for i in range(self._n_layers)])
        else:
            self._mmi_layer_odd = nn.ModuleList([MMILayerMatrix_Odd(n_inputs=n_inputs) for _ in range(self._n_layers)])
        # IN OUT
        self._ht_layer_in = HeaterLayerMatrix_Full(n_inputs=n_inputs)
        self._ht_layer_out = HeaterLayerMatrix_Full(n_inputs=n_inputs)

    def forward(self):
        id_ini = torch.eye(self._n_inputs, device=device)      # Identity matrix
        arct_matrix = torch.complex(id_ini, torch.zeros_like(id_ini, device=device))
        arct_matrix = self._ht_layer_in(arct_matrix)
        for i in range(self._n_layers//2):
            arct_matrix = self._mmi_layer_even[2*i](arct_matrix)
            arct_matrix = self._ht_layer_full[2*i](arct_matrix)
            arct_matrix = self._mmi_layer_even[2*i+1](arct_matrix)
            arct_matrix = self._mmi_layer_odd[2*i](arct_matrix)
            arct_matrix = self._ht_layer_full[2*i+1](arct_matrix)
            arct_matrix = self._mmi_layer_odd[2*i+1](arct_matrix)
        arct_matrix = self._ht_layer_out(arct_matrix)
        return arct_matrix



# Fldzhyan Architecture -------------------------------------------------------------------------------------------
class Fldzhyan_Arct(nn.Module):
    r"""
    Fldzhyan architecture:
        0__[]__  __________[]__  __________[]__  __________[]__  __________[]__0
               \/              \/              \/              \/
        1______/\__[]__  ______/\__[]__  ______/\__[]__  ______/\__[]__  __[]__1
                       \/              \/              \/              \/
        2__[]__  ______/\__[]__  ______/\__[]__  ______/\__[]__  ______/\__[]__2
               \/              \/              \/              \/
        3______/\______________/\______________/\______________/\__________[]__3

        with:
            0__[]__1 = Phase Shifter

            0__  __0
               \/    =  MMI
            1__/\__1
    """
    def __init__(self,
                 n_inputs: int,
                 mmi_i_losses_mtx_even: torch.Tensor = None,
                 mmi_i_losses_mtx_odd: torch.Tensor = None,
                 mmi_imbalances_mtx_even: torch.Tensor = None,
                 mmi_imbalances_mtx_odd: torch.Tensor = None):
        super(Fldzhyan_Arct, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        self._n_inputs = n_inputs
        self._n_layers = n_inputs
        
        # EVEN
        self._ht_layer_even = nn.ModuleList([HeaterLayerMatrix_Even(n_inputs=n_inputs) for _ in range(self._n_layers)])
        if (mmi_i_losses_mtx_even is not None) and (mmi_imbalances_mtx_even is not None):
            self._mmi_layer_even = nn.ModuleList([MMILayerMatrix_Even(
                n_inputs=n_inputs,
                mmi_i_losses=mmi_i_losses_mtx_even[i],
                mmi_imbalances=mmi_imbalances_mtx_even[i])
                for i in range(self._n_layers)])
        else:
            self._mmi_layer_even = nn.ModuleList([MMILayerMatrix_Even(n_inputs=n_inputs) for _ in range(self._n_layers)])
        # ODD
        self._ht_layer_odd = nn.ModuleList([HeaterLayerMatrix_Odd(n_inputs=n_inputs) for _ in range(self._n_layers)])
        if (mmi_i_losses_mtx_odd is not None) and (mmi_imbalances_mtx_odd is not None):
            self._mmi_layer_odd = nn.ModuleList([MMILayerMatrix_Odd(
                n_inputs=n_inputs,
                mmi_i_losses=mmi_i_losses_mtx_odd[i],
                mmi_imbalances=mmi_imbalances_mtx_odd[i])
                for i in range(self._n_layers)])
        self._mmi_layer_odd = nn.ModuleList([MMILayerMatrix_Odd(n_inputs=n_inputs) for _ in range(self._n_layers)])
        # OUT
        self._ht_layer_out = HeaterLayerMatrix_Full(n_inputs=n_inputs)

    def forward(self):
        id_ini = torch.eye(self._n_inputs, device=device)      # Identity matrix
        arct_matrix = torch.complex(id_ini, torch.zeros_like(id_ini, device=device))
        for i in range(self._n_layers):
            arct_matrix = self._mmi_layer_even[i](arct_matrix)
            arct_matrix = self._ht_layer_even[i](arct_matrix)
            arct_matrix = self._mmi_layer_odd[i](arct_matrix)
            arct_matrix = self._ht_layer_odd[i](arct_matrix)
        arct_matrix = self._ht_layer_out(arct_matrix)
        return arct_matrix



# Fldzhyan Bell Architecture --------------------------------------------------------------------------------------
class FldzhyanBell_Arct(nn.Module):
    r"""
    Fldzhyan architecture:
        0__[]__  __[]__________  __[]__________  __[]__________  __[]______[]__0
               \/              \/              \/              \/
        1__[]__/\__[]__  ______/\__[]__  ______/\__[]__  ______/\__[]__  __[]__1
                       \/              \/              \/              \/
        2__[]__  __[]__/\______  __[]__/\______  __[]__/\______  __[]__/\__[]__2
               \/              \/              \/              \/
        3__[]__/\__[]__________/\__[]__________/\__[]__________/\__[]______[]__3

        with:
            0__[]__1 = Phase Shifter

            0__  __0
               \/    =  MMI
            1__/\__1
    """
    def __init__(self,
                 n_inputs: int,
                 mmi_i_losses_mtx_even: torch.Tensor = None,
                 mmi_i_losses_mtx_odd: torch.Tensor = None,
                 mmi_imbalances_mtx_even: torch.Tensor = None,
                 mmi_imbalances_mtx_odd: torch.Tensor = None):
        super(FldzhyanBell_Arct, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        self._n_inputs = n_inputs
        self._n_layers = n_inputs
        
        self._ht_layer_full = nn.ModuleList([HeaterLayerMatrix_Full(n_inputs=n_inputs) for _ in range(self._n_layers)])
        # EVEN
        if (mmi_i_losses_mtx_even is not None) and (mmi_imbalances_mtx_even is not None):
            self._mmi_layer_even = nn.ModuleList([MMILayerMatrix_Even(
                n_inputs=n_inputs,
                mmi_i_losses=mmi_i_losses_mtx_even[i],
                mmi_imbalances=mmi_imbalances_mtx_even[i])
                for i in range(self._n_layers)])
        else:
            self._mmi_layer_even = nn.ModuleList([MMILayerMatrix_Even(n_inputs=n_inputs) for _ in range(self._n_layers)])
        # ODD
        if (mmi_i_losses_mtx_odd is not None) and (mmi_imbalances_mtx_odd is not None):
            self._mmi_layer_odd = nn.ModuleList([MMILayerMatrix_Odd(
                n_inputs=n_inputs,
                mmi_i_losses=mmi_i_losses_mtx_odd[i],
                mmi_imbalances=mmi_imbalances_mtx_odd[i])
                for i in range(self._n_layers)])
        self._mmi_layer_odd = nn.ModuleList([MMILayerMatrix_Odd(n_inputs=n_inputs) for _ in range(self._n_layers)])
        # IN OUT
        self._ht_layer_in = HeaterLayerMatrix_Full(n_inputs=n_inputs)
        self._ht_layer_out = HeaterLayerMatrix_Full(n_inputs=n_inputs)

    def forward(self):
        id_ini = torch.eye(self._n_inputs, device=device)      # Identity matrix
        arct_matrix = torch.complex(id_ini, torch.zeros_like(id_ini, device=device))
        arct_matrix = self._ht_layer_in(arct_matrix)
        for i in range(self._n_layers):
            arct_matrix = self._mmi_layer_even[i](arct_matrix)
            arct_matrix = self._ht_layer_full[i](arct_matrix)
            arct_matrix = self._mmi_layer_odd[i](arct_matrix)
        arct_matrix = self._ht_layer_out(arct_matrix)
        return arct_matrix



# Fldzhyan Bell Half Architecture ---------------------------------------------------------------------------------
class FldzhyanBellHalf_Arct(nn.Module):
    r"""
    Fldzhyan architecture:
        0__[]__  __[]__________  __[]______[]__0
               \/              \/
        1__[]__/\__[]__  ______/\__[]__  __[]__1
                       \/              \/
        2__[]__  __[]__/\______  __[]__/\__[]__2
               \/              \/
        3__[]__/\__[]__________/\__[]______[]__3

        with:
            0__[]__1 = Phase Shifter

            0__  __0
               \/    =  MMI
            1__/\__1
    """
    def __init__(self,
                 n_inputs: int,
                 mmi_i_losses_mtx_even: torch.Tensor = None,
                 mmi_i_losses_mtx_odd: torch.Tensor = None,
                 mmi_imbalances_mtx_even: torch.Tensor = None,
                 mmi_imbalances_mtx_odd: torch.Tensor = None):
        super(FldzhyanBellHalf_Arct, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        self._n_inputs = n_inputs
        self._n_layers = n_inputs//2
        
        self._ht_layer_full = nn.ModuleList([HeaterLayerMatrix_Full(n_inputs=n_inputs) for _ in range(self._n_layers)])
        # EVEN
        if (mmi_i_losses_mtx_even is not None) and (mmi_imbalances_mtx_even is not None):
            self._mmi_layer_even = nn.ModuleList([MMILayerMatrix_Even(
                n_inputs=n_inputs,
                mmi_i_losses=mmi_i_losses_mtx_even[i],
                mmi_imbalances=mmi_imbalances_mtx_even[i])
                for i in range(self._n_layers)])
        else:
            self._mmi_layer_even = nn.ModuleList([MMILayerMatrix_Even(n_inputs=n_inputs) for _ in range(self._n_layers)])
        # ODD
        if (mmi_i_losses_mtx_odd is not None) and (mmi_imbalances_mtx_odd is not None):
            self._mmi_layer_odd = nn.ModuleList([MMILayerMatrix_Odd(
                n_inputs=n_inputs,
                mmi_i_losses=mmi_i_losses_mtx_odd[i],
                mmi_imbalances=mmi_imbalances_mtx_odd[i])
                for i in range(self._n_layers)])
        else:
            self._mmi_layer_odd = nn.ModuleList([MMILayerMatrix_Odd(n_inputs=n_inputs) for _ in range(self._n_layers)])
        # IN OUT
        self._ht_layer_in = HeaterLayerMatrix_Full(n_inputs=n_inputs)
        self._ht_layer_out = HeaterLayerMatrix_Full(n_inputs=n_inputs)

    def forward(self):
        id_ini = torch.eye(self._n_inputs, device=device)      # Identity matrix
        arct_matrix = torch.complex(id_ini, torch.zeros_like(id_ini, device=device))
        arct_matrix = self._ht_layer_in(arct_matrix)
        for i in range(self._n_layers):
            arct_matrix = self._mmi_layer_even[i](arct_matrix)
            arct_matrix = self._ht_layer_full[i](arct_matrix)
            arct_matrix = self._mmi_layer_odd[i](arct_matrix)
        arct_matrix = self._ht_layer_out(arct_matrix)
        return arct_matrix



# NEUROPULS Architecture ------------------------------------------------------------------------------------------
class NEUROPULS_Arct(nn.Module):
    r"""
    NEUROPULS architecture:
        0__[]__  __[]__  ______   __[]__  __[]__  __[]__0
               \/      \/      \-/      \/      \/
        1______/\______/\______   ______/\______/\__[]__1
                               \-/
        2__[]__  __[]__  ______/-\__[]__  __[]__  __[]__2
               \/      \/               \/      \/
        3______/\______/\______/-\______/\______/\__[]__3

        with:
            0__[]__1 = Phase Shifter

            0__  __0
               \/    =  MMI
            1__/\__1
            
            0__   __0
               \-/   =  Crossing
            1__/-\__1
    """
    def __init__(self,
                 n_inputs: int,
                 mmi_i_losses_mtx_even: torch.Tensor = None,
                 mmi_imbalances_mtx_even: torch.Tensor = None,
                 crossing_i_losses_mtx_odd: torch.Tensor = None,
                 crossing_crosstalks_mtx_odd: torch.Tensor = None):
        super(NEUROPULS_Arct, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        self._n_inputs = n_inputs
        self._n_layers_ht = n_inputs
        self._n_layers_mmi = n_inputs
        self._n_layers_crossing = n_inputs//2 - 1
        
        # EVEN
        self._ht_layer_even = nn.ModuleList([HeaterLayerMatrix_Even(n_inputs=n_inputs) for _ in range(self._n_layers_ht)])
        if (mmi_i_losses_mtx_even is not None) and (mmi_imbalances_mtx_even is not None):
            self._mmi_layer_even = nn.ModuleList([MMILayerMatrix_Even(
                n_inputs=n_inputs,
                mmi_i_losses=mmi_i_losses_mtx_even[i],
                mmi_imbalances=mmi_imbalances_mtx_even[i])
                for i in range(self._n_layers_mmi)])
        else:
            self._mmi_layer_even = nn.ModuleList([MMILayerMatrix_Even(n_inputs=n_inputs) for _ in range(self._n_layers_mmi)])
        # ODD
        if (crossing_i_losses_mtx_odd is not None) and (crossing_crosstalks_mtx_odd is not None):
            self._crossing_layer_odd = nn.ModuleList([CrossingLayerMatrix_Odd(
                n_inputs=n_inputs,
                crossing_i_losses=crossing_i_losses_mtx_odd[i],
                crossing_crosstalks=crossing_crosstalks_mtx_odd[i])
                for i in range(self._n_layers_crossing)])
        else:
            self._crossing_layer_odd = nn.ModuleList([CrossingLayerMatrix_Odd(n_inputs=n_inputs) for _ in range(self._n_layers_crossing)])
        
        # OUT
        self._ht_layer_out = HeaterLayerMatrix_Full(n_inputs=n_inputs)

    def forward(self):
        id_ini = torch.eye(self._n_inputs, device=device)      # Identity matrix
        arct_matrix = torch.complex(id_ini, torch.zeros_like(id_ini, device=device))
        for i in range(self._n_layers_ht//2):
            arct_matrix = self._ht_layer_even[2*i](arct_matrix)
            arct_matrix = self._mmi_layer_even[2*i](arct_matrix)
            arct_matrix = self._ht_layer_even[2*i+1](arct_matrix)
            arct_matrix = self._mmi_layer_even[2*i+1](arct_matrix)
            if i < self._n_layers_ht//2 - 1:
                self._crossing_layer_odd[i](arct_matrix)
        arct_matrix = self._ht_layer_out(arct_matrix)
        return arct_matrix



# NEUROPULS Bell Architecture -------------------------------------------------------------------------------------
class NEUROPULSBell_Arct(nn.Module):
    r"""
    NEUROPULS Bell architecture:
        0__[]__  __[]__  ______   ______  __[]__  __[]__0
               \/      \/      \-/      \/      \/
        1__[]__/\__[]__/\______   ______/\__[]__/\__[]__1
                               \-/
        2__[]__  __[]__  ______/-\______  __[]__  __[]__2
               \/      \/               \/      \/
        3__[]__/\__[]__/\______/-\______/\__[]__/\__[]__3

        with:
            0__[]__1 = Phase Shifter

            0__  __0
               \/    =  MMI
            1__/\__1
            
            0__   __0
               \-/   =  Crossing
            1__/-\__1
    """
    def __init__(self,
                 n_inputs: int,
                 mmi_i_losses_mtx_even: torch.Tensor = None,
                 mmi_imbalances_mtx_even: torch.Tensor = None,
                 crossing_i_losses_mtx_odd: torch.Tensor = None,
                 crossing_crosstalks_mtx_odd: torch.Tensor = None):
        super(NEUROPULSBell_Arct, self).__init__()
        if n_inputs%2 == 1: raise Exception('n_inputs is odd!!! NONONO, put it even!!!')
        self._n_inputs = n_inputs
        self._n_layers_ht = n_inputs//2
        self._n_layers_mmi = n_inputs
        self._n_layers_crossing = n_inputs//2 - 1
        
        self._ht_layer_full = nn.ModuleList([HeaterLayerMatrix_Full(n_inputs=n_inputs) for _ in range(self._n_layers_ht)])
        # EVEN
        if (mmi_i_losses_mtx_even is not None) and (mmi_imbalances_mtx_even is not None):
            self._mmi_layer_even = nn.ModuleList([MMILayerMatrix_Even(
                n_inputs=n_inputs,
                mmi_i_losses=mmi_i_losses_mtx_even[i],
                mmi_imbalances=mmi_imbalances_mtx_even[i])
                for i in range(self._n_layers_mmi)])
        else:
            self._mmi_layer_even = nn.ModuleList([MMILayerMatrix_Even(n_inputs=n_inputs) for _ in range(self._n_layers_mmi)])
        # ODD
        if (crossing_i_losses_mtx_odd is not None) and (crossing_crosstalks_mtx_odd is not None):
            self._crossing_layer_odd = nn.ModuleList([CrossingLayerMatrix_Odd(
                n_inputs=n_inputs,
                crossing_i_losses=crossing_i_losses_mtx_odd[i],
                crossing_crosstalks=crossing_crosstalks_mtx_odd[i])
                for i in range(self._n_layers_crossing)])
        else:
            self._crossing_layer_odd = nn.ModuleList([CrossingLayerMatrix_Odd(n_inputs=n_inputs) for _ in range(self._n_layers_crossing)])
        
        # IN OUT
        self._ht_layer_in = HeaterLayerMatrix_Full(n_inputs=n_inputs)
        self._ht_layer_out = HeaterLayerMatrix_Full(n_inputs=n_inputs)

    def forward(self):
        id_ini = torch.eye(self._n_inputs, device=device)      # Identity matrix
        arct_matrix = torch.complex(id_ini, torch.zeros_like(id_ini, device=device))
        arct_matrix = self._ht_layer_in(arct_matrix)
        for i in range(self._n_layers_ht):
            arct_matrix = self._mmi_layer_even[2*i](arct_matrix)
            arct_matrix = self._ht_layer_full[i](arct_matrix)
            arct_matrix = self._mmi_layer_even[2*i+1](arct_matrix)
            if i < self._n_layers_ht - 1:
                self._crossing_layer_odd[i](arct_matrix)
        arct_matrix = self._ht_layer_out(arct_matrix)
        return arct_matrix


