# TO DO:
# 1- how to set the temporal kernel_size? The temporal sampling rate of the input and the kernel should be the same
# 2- how to set the kernel_size?

import os, sys
import pdb

import numpy as np
import scipy as sp
from sympy.abc import x as symbolic_x
from sympy.abc import y as symbolic_y
import yaml

import torch
import torch.nn as nn
import pandas as pd

sys.path.append('../bmtk')
from bmtk.simulator.filternet.lgnmodel.temporalfilter import TemporalFilterCosineBump
from bmtk.simulator.filternet.lgnmodel.spatialfilter import GaussianSpatialFilter
from bmtk.simulator.filternet.lgnmodel.linearfilter import SpatioTemporalFilter
from bmtk.simulator.filternet.lgnmodel.cellmodel import TwoSubfieldLinearCell
from bmtk.simulator.filternet.lgnmodel.transferfunction import MultiTransferFunction


class Conv3dLGN(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, cell_type, conv_type = 'dom'):
        padding = tuple(x//2 for x in kernel_size)
        super(Conv3dLGN,self).__init__(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size, padding=padding)
        self._set_weights(cell_type, out_channels, conv_type)   
    
    def _set_weights(self, cell_type, num_cells, conv_type):
        
        kernels = self._make_kernels(cell_type, num_cells)    
        self.weight = nn.Parameter(kernels[conv_type], requires_grad=False) #[:,:,:,:,:]

    def _load_param_values(self):

        param_file_path = '/home/mila/b/bakhtias/Project-Codes/CPC/backbone/lgn_full_col_cells_3.csv'
        lgn_types_file_path = '/home/mila/b/bakhtias/Project-Codes/CPC/backbone/lgn_full_col_cell_models_3.csv'

        param_table = pd.read_csv(param_file_path, sep=' ')
        lgn_types_table = pd.read_csv(lgn_types_file_path)

        return param_table, lgn_types_table

    def _make_kernels(self,cell_type, num_cells):

        param_table, lgn_types_table = self._load_param_values()

        all_spatial_sizes = param_table['spatial_size'][param_table['model_id']==cell_type]
        all_kpeaks_dom_0s = param_table['kpeaks_dom_0'][param_table['model_id']==cell_type]
        all_kpeaks_dom_1s = param_table['kpeaks_dom_1'][param_table['model_id']==cell_type]
        all_weight_dom_0s = param_table['weight_dom_0'][param_table['model_id']==cell_type]
        all_weight_dom_1s = param_table['weight_dom_1'][param_table['model_id']==cell_type]
        all_delay_dom_0s = param_table['delay_dom_0'][param_table['model_id']==cell_type]
        all_delay_dom_1s = param_table['delay_dom_1'][param_table['model_id']==cell_type]
        all_kpeaks_non_dom_0s = param_table['kpeaks_non_dom_0'][param_table['model_id']==cell_type]
        all_kpeaks_non_dom_1s = param_table['kpeaks_non_dom_1'][param_table['model_id']==cell_type]
        all_weight_non_dom_0s = param_table['weight_non_dom_0'][param_table['model_id']==cell_type]
        all_weight_non_dom_1s = param_table['weight_non_dom_1'][param_table['model_id']==cell_type]
        all_delay_non_dom_0s = param_table['delay_non_dom_0'][param_table['model_id']==cell_type]
        all_delay_non_dom_1s = param_table['delay_non_dom_1'][param_table['model_id']==cell_type]
        all_sf_seps = param_table['sf_sep'][param_table['model_id']==cell_type]
        all_angles = param_table['tuning_angle'][param_table['model_id']==cell_type]

        # this needs to be corrected for sONsOFF/sONtOFF cells
        if (('sOFF' in cell_type) or ('tOFF' in cell_type)) and (cell_type != 'sONsOFF_001') and (cell_type != 'sONtOFF_001'):
            amplitude = -1.0
        elif (cell_type == 'sONsOFF_001') or (cell_type == 'sONtOFF_001'):
            amplitude = 1.0
            amplitude_2 = -1.0
        else:
            amplitude = 1.0
            

        kdom_data = torch.empty((num_cells,3,*self.kernel_size))
        k_dom_nondom_data = torch.empty((num_cells,3,*self.kernel_size))
        kernels = dict()

        for cellcount in range(0,num_cells):
            
            sampled_cell_idx = int(torch.randint(low=min(all_kpeaks_dom_0s.keys()),high=max(all_kpeaks_dom_0s.keys()),size=(1,1)))

            Tdom = TemporalFilterCosineBump(weights=(all_weight_dom_0s[sampled_cell_idx],all_weight_dom_1s[sampled_cell_idx]), 
                                            kpeaks=(all_kpeaks_dom_0s[sampled_cell_idx],all_kpeaks_dom_1s[sampled_cell_idx]), 
                                            delays=(all_delay_dom_0s[sampled_cell_idx],all_delay_dom_1s[sampled_cell_idx]))
            

            this_sigma = all_spatial_sizes[sampled_cell_idx]
            this_sf_sep = all_sf_seps[sampled_cell_idx]
            this_angle = all_angles[sampled_cell_idx]

            Sdom = GaussianSpatialFilter(translate=(0.0, 0.0), 
                                        sigma=(this_sigma, this_sigma), 
                                        rotation=0, 
                                        origin='center')

            Kerneldom = SpatioTemporalFilter(spatial_filter = Sdom, temporal_filter = Tdom, amplitude=amplitude)
            # Kerneldom.show_temporal_filter(show=True)
            # Kerneldom.show_spatial_filter(row_range=range(0,10),col_range=range(0,10),show=True)
            
            
            kdom = Kerneldom.get_spatiotemporal_kernel(row_range=range(0,self.kernel_size[1]),col_range=range(0,self.kernel_size[2]))
            temporal_ds_rate = (kdom.full().shape[0]-2)//(self.kernel_size[0]-1)

            if cell_type != 'sONsOFF_001' and cell_type != 'sONtOFF_001':
                kdom_data[cellcount,:,:,:,:] = torch.Tensor(kdom.full())[::temporal_ds_rate,:,:].repeat([3,1,1,1])
                kernels['dom'] = kdom_data

            elif cell_type == 'sONsOFF_001' or cell_type == 'sONtOFF_001':
                Tnondom = TemporalFilterCosineBump(weights=(all_weight_non_dom_0s[sampled_cell_idx],all_weight_non_dom_1s[sampled_cell_idx]), 
                                            kpeaks=(all_kpeaks_non_dom_0s[sampled_cell_idx],all_kpeaks_non_dom_1s[sampled_cell_idx]), 
                                            delays=(all_delay_non_dom_0s[sampled_cell_idx],all_delay_non_dom_1s[sampled_cell_idx]))

            
                Snondom = GaussianSpatialFilter(translate=(0.0, 0.0), 
                                            sigma=(this_sigma, this_sigma), 
                                            rotation=0, 
                                            origin='center')
                
                Kernelnondom = SpatioTemporalFilter(spatial_filter = Snondom, temporal_filter = Tnondom, amplitude=amplitude_2)

                KernelOnOff = TwoSubfieldLinearCell(dominant_filter = Kerneldom, 
                                                    nondominant_filter = Kernelnondom, 
                                                    subfield_separation = this_sf_sep,
                                                    onoff_axis_angle = this_angle,
                                                    dominant_subfield_location = (0, 0),
                                                    transfer_function = MultiTransferFunction((symbolic_x, symbolic_y),'Heaviside(x)*(x)+Heaviside(y)*(y)'))
                k_dom_nondom = KernelOnOff.get_spatiotemporal_kernel(row_range=range(0,self.kernel_size[1]),col_range=range(0,self.kernel_size[2]))
                
                k_dom_nondom_data[cellcount,:,:,:,:] = torch.Tensor(k_dom_nondom.full())[::temporal_ds_rate,:,:].repeat([3,1,1,1]) 
                kernels['dom_nondom'] = k_dom_nondom_data

        return kernels

class Conv3dLGN_layer(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(Conv3dLGN_layer, self).__init__()

        cell_specs = yaml.load(open('/home/mila/b/bakhtias/Project-Codes/CPC/backbone/cell_specs.yaml'), Loader=yaml.FullLoader)
        self.cell_types = cell_specs['cell_types']
        self.num_cells_per_type = cell_specs['num_cells']

        self.Convs = nn.ModuleDict()
        self.out_channels = 0
        for cell_type, num_cells in zip(self.cell_types,self.num_cells_per_type):
            self.out_channels += num_cells
            if cell_type != 'sONsOFF_001' and cell_type != 'sONtOFF_001':  
                self.Convs[cell_type+'_dom'] = Conv3dLGN(in_channels=in_channels, out_channels=num_cells, kernel_size=kernel_size, cell_type=cell_type)
            else:
                self.Convs[cell_type+'_dom_nondom'] = Conv3dLGN(in_channels=in_channels, out_channels=num_cells, kernel_size=kernel_size, cell_type=cell_type, conv_type='dom_nondom')
        self.ReLU = nn.ReLU()

    def forward(self, x):
        B, C, D, W, H = x.shape
        out = torch.empty((B, self.out_channels, D, W, H),device='cuda')
        i = 0
        for cell_type, num_cells in zip(self.cell_types, self.num_cells_per_type):
            if cell_type != 'sONsOFF_001' and cell_type != 'sONtOFF_001':
                out[:,i:(i+num_cells),:] = (self.Convs[cell_type+'_dom'](x))
            else:
                out[:,i:(i+num_cells),:] = (self.Convs[cell_type+'_dom_nondom'](x))
            i += num_cells
        
        return out


if  __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    import wandb

    data = np.load('allen_movie_one.npy').astype('float64')
    data_norm = ((data * 2) - np.max(data)) / np.max(data)
    print(data.shape,np.max(data_norm),np.min(data_norm))
    
    wandb.init(project="test-convLGN")
    kernel_size = (11,11,11)
    wandb.config.kernel_sizes = kernel_size
    
    t0 = time.time()
    cuda = torch.device('cuda')
    
#     convmodel = Conv3dLGN(in_channels = 3, out_channels = 1, kernel_size = (10,10,10), cell_type = 'sON_TF2', conv_type = 'dom').to(cuda)
    LGN_layer = Conv3dLGN_layer(in_channels=3, kernel_size= kernel_size)
    LGN_layer = LGN_layer.to(cuda)
    t1 = time.time()
    x = torch.Tensor(data[::1,:,:]).unsqueeze_(0).unsqueeze_(0).repeat([1,3,1,1,1]).to(cuda)
#     x = torch.rand((1, 3, 20, 64, 64)).to('cuda')
#     x = torch.Tensor(data[::30,:,:]).view((1,1,*data.shape)).to('cuda')
    out = LGN_layer(x).detach().numpy()
    t2 = time.time()
    
    print(LGN_layer.cell_types) 
    print(LGN_layer.Convs)
    for cell in LGN_layer.cell_types:
        for i in range(LGN_layer.Convs[f"{cell}_dom"].weight.shape[2]):
            I = LGN_layer.Convs[f"{cell}_dom"].weight[0,0,i,:,:].detach().cpu().numpy()
            wandb.log({f"spatial kernel {cell}" : [wandb.Image(I)]})
        
        T = LGN_layer.Convs[f"{cell}_dom"].weight[0,0,:,5,5].detach().cpu().numpy()
        data = [[x, y] for (x, y) in zip(np.arange(len(T)), T)]
        table = wandb.Table(data=data, columns = ["x", "y"])
        wandb.log({f"temporal kernel {cell}" : wandb.plot.line(table, "x", "y", title=cell)})

#     wandb.alert()
    
    
#     with open('LGN_allen_movie_one.npy','wb') as f:
#         np.save(f,out)
#     pdb.set_trace()
