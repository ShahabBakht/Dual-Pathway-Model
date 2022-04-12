import sys
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../backbone')
from select_backbone import select_resnet, select_mousenet, select_simmousenet, select_monkeynet
from convrnn import ConvGRU


class DPC_Plus(nn.Module):
    '''DPC with RNN'''
    def __init__(self, sample_size, num_seq=8, seq_len=5, pred_step=3, network='monkeynet', heads=['heading','obj'], paths=['heading','obj']): #['heading','obj']
        super(DPC_Plus, self).__init__()
#         torch.cuda.manual_seed(233) #233
        print('Using DPC-RNN model')
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.pred_step = pred_step
        self.heads = heads
        self.paths = paths
        
        if network == 'vgg' or network == 'mousenet' or network == 'simmousenet' or network == 'monkeynet':
            self.last_duration = seq_len
        else:
            self.last_duration = int(math.ceil(seq_len / 4))
        
        if network == 'resnet0':
            self.last_size = int(math.ceil(sample_size / 8)) #8
            self.pool_size = 1
        elif network == 'mousenet':
            self.last_size = 16
            self.pool_size = 2 # (2 for all readout, 4 for VISp5 readout)
        elif network == 'simmousenet':
            self.last_size = 16
            self.pool_size = 1
        elif network == 'monkeynet':
            self.last_size = 16
            self.pool_size = 1
        else:
            self.last_size = int(math.ceil(sample_size / 32))
            self.pool_size = 1
            
            
        print('final feature map has size %dx%d' % (self.last_size, self.last_size))
        if network == 'mousenet':
            self.backbone, self.param = select_mousenet()
        elif network == 'simmousenet':
            self.backbone, self.param = select_simmousenet(hp)
        elif network == 'monkeynet':
            self.backbone, self.param = select_monkeynet()
        else:
            self.backbone, self.param = select_resnet(network, track_running_stats=False)
            
        self.param['num_layers'] = 1 # param for GRU
        self.param['hidden_size'] = self.param['feature_size'] # param for GRU

        self.agg = ConvGRU(input_size=self.param['feature_size'],
                               hidden_size=self.param['hidden_size'],
                               kernel_size=1,
                               num_layers=self.param['num_layers'])
        self.network_pred = nn.Sequential(
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0)
                                )
        self.mask = None
        self.relu = nn.ReLU(inplace=False)
        
        if 'heading' in self.heads:
            self.linear1_1 = nn.Linear(1*self.backbone.path1.resblocks_out_channels,64)
            self.linear1_2 = nn.Linear(1*self.backbone.path1.resblocks_out_channels,64)
            self.linear2_1 = nn.Linear(64,2)
            self.linear2_2 = nn.Linear(64,1)
        if 'obj' in self.heads:
            self.linear_obj = nn.Linear(self.backbone.path1.resblocks_out_channels,73)
        
        self._initialize_weights(self.agg)
        self._initialize_weights(self.network_pred)

    def forward(self, block):
        # block: [B, N, C, SL, W, H]
        ### extract feature ###
        (B, N, C, SL, H, W) = block.shape
        block = block.view(B*N, C, SL, H, W)
        
        ### predictive part
        feature, feature_1, feature_2 = self.backbone(block) #
         
        feature = F.avg_pool3d(feature, (self.last_duration, self.pool_size, self.pool_size), stride=(1, self.pool_size, self.pool_size))
        feature_inf_all = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # before ReLU, (-inf, +inf)
        feature = self.relu(feature) # [0, +inf)
        feature = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # [B,N,D,6,6], [0, +inf)
        feature_inf = feature_inf_all[:, N-self.pred_step::, :].contiguous()
        del feature_inf_all

        ### aggregate, predict future ###
        _, hidden = self.agg(feature[:, 0:N-self.pred_step, :].contiguous())
        hidden = hidden[:,-1,:] # after tanh, (-1,1). get the hidden state of last layer, last time step
        
        pred = []
        for i in range(self.pred_step):
            # sequentially pred future
            p_tmp = self.network_pred(hidden)
            pred.append(p_tmp)
            _, hidden = self.agg(self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
            hidden = hidden[:,-1,:]
        pred = torch.stack(pred, 1) # B, pred_step, xxx
        del hidden


        ### Get similarity score ###
        # pred: [B, pred_step, D, last_size, last_size]
        # GT: [B, N, D, last_size, last_size]
        N = self.pred_step
        # dot product D dimension in pred-GT pair, get a 6d tensor. First 3 dims are from pred, last 3 dims are from GT. 
        pred = pred.permute(0,1,3,4,2).contiguous().view(B*self.pred_step*self.last_size**2, self.param['feature_size'])
        feature_inf = feature_inf.permute(0,1,3,4,2).contiguous().view(B*N*self.last_size**2, self.param['feature_size']).transpose(0,1)
        score = torch.matmul(pred, feature_inf).view(B, self.pred_step, self.last_size**2, B, N, self.last_size**2)
        del feature_inf, pred

        if self.mask is None: # only compute mask once
            # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
            mask = torch.zeros((B, self.pred_step, self.last_size**2, B, N, self.last_size**2), dtype=torch.int8, requires_grad=False).detach().cuda()
            mask[torch.arange(B), :, :, torch.arange(B), :, :] = -3 # spatial neg
            for k in range(B):
                mask[k, :, torch.arange(self.last_size**2), k, :, torch.arange(self.last_size**2)] = -1 # temporal neg
            tmp = mask.permute(0, 2, 1, 3, 5, 4).contiguous().view(B*self.last_size**2, self.pred_step, B*self.last_size**2, N)
            for j in range(B*self.last_size**2):
                tmp[j, torch.arange(self.pred_step), j, torch.arange(N-self.pred_step, N)] = 1 # pos
            mask = tmp.view(B, self.last_size**2, self.pred_step, B, self.last_size**2, N).permute(0,2,1,3,5,4)
            self.mask = mask
        
        
        ### heading part ###
        N = block.shape[0]//B
        (BN, Cout1, SL, Hout1, Wout1) = feature_2.shape
        if self.paths[0] == 'heading' and self.paths[1] == 'obj':
            # print('heading is first')
            feature_hd = feature_1.view(B, N, Cout1, SL, Hout1, Wout1).permute(0,2,1,3,4,5).contiguous().view(B, Cout1, N*SL, Hout1, Wout1) #
            feature_obj = feature_2.view(B, N, Cout1, SL, Hout1, Wout1).permute(0,2,1,3,4,5).contiguous().view(B, Cout1, N*SL, Hout1, Wout1) #
        elif self.paths[0] == 'obj' and self.paths[1] == 'heading':
            # print('obj is first')
            feature_hd = feature_2.view(B, N, Cout1, SL, Hout1, Wout1).permute(0,2,1,3,4,5).contiguous().view(B, Cout1, N*SL, Hout1, Wout1) #
            feature_obj = feature_1.view(B, N, Cout1, SL, Hout1, Wout1).permute(0,2,1,3,4,5).contiguous().view(B, Cout1, N*SL, Hout1, Wout1) #
    
#         feature_cat = torch.cat((feature_1,feature_2), dim=1)
#         (BN, Cout1, SL, Hout1, Wout1) = feature_cat.shape
#         feature_hd = feature_cat.view(B, N, Cout1, SL, Hout1, Wout1).permute(0,2,1,3,4,5).contiguous().view(B, Cout1, N*SL, Hout1, Wout1)

        del block

        if ('obj' in self.heads) and ('heading' not in self.heads):
#             feature_hd = self.backbone.path1(feature_hd)
            feature_obj = nn.functional.avg_pool3d(feature_obj, kernel_size = (feature_hd.shape[2], feature_hd.shape[3], feature_hd.shape[4])).squeeze()
            y_obj = self.linear_obj(feature_obj)
            
            return [score, self.mask], y_obj
        
        elif ('heading' in self.heads) and ('obj' not in self.heads):
#             feature_hd = self.backbone.path1(feature_hd)
            feature_hd = nn.functional.avg_pool3d(feature_hd, kernel_size = (feature_hd.shape[2], feature_hd.shape[3], feature_hd.shape[4])).squeeze()
            feature_hd_1 = self.linear1_1(feature_hd)
            feature_hd_2 = self.linear1_2(feature_hd)
            feature_hd_1 = self.relu(feature_hd_1)
            feature_hd_2 = self.relu(feature_hd_2)
            y_hd_1 = self.linear2_1(feature_hd_1)
            y_hd_2 = self.linear2_2(feature_hd_2)
            y_hd = torch.cat((y_hd_1,y_hd_2),1)
        
            return [score, self.mask], y_hd
        
        elif ('heading' in self.heads) and ('obj' in self.heads):
            
            feature_obj = nn.functional.avg_pool3d(feature_obj, kernel_size = (feature_hd.shape[2], feature_hd.shape[3], feature_hd.shape[4])).squeeze()
            y_obj = self.linear_obj(feature_obj)
            
            feature_hd = nn.functional.avg_pool3d(feature_hd, kernel_size = (feature_hd.shape[2], feature_hd.shape[3], feature_hd.shape[4])).squeeze()
            feature_hd_1 = self.linear1_1(feature_hd)
            feature_hd_2 = self.linear1_2(feature_hd)
            feature_hd_1 = self.relu(feature_hd_1)
            feature_hd_2 = self.relu(feature_hd_2)
            y_hd_1 = self.linear2_1(feature_hd_1)
            y_hd_2 = self.linear2_2(feature_hd_2)
            y_hd = torch.cat((y_hd_1,y_hd_2),1)
            
            return [score, self.mask], y_hd, y_obj
        
        

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself

    def reset_mask(self):
        self.mask = None
    
    

if __name__ == '__main__':
    x = torch.rand((20, 8, 3, 5, 64, 64)).to('cuda')
    model = DPC_Plus(sample_size=64, 
                        num_seq=8, 
                        seq_len=5, 
                        network='monkeynet', 
                        pred_step=3).to('cuda')
    [score, mask], y_hd = model(x)
    print(score.shape, mask.shape, y_hd.shape)
