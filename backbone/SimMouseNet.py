import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import networkx as nx
import yaml


from SimMouseNet_network import Network
from convrnn import ConvGRU


class SimMouseNet(nn.Module):
    def __init__(self):
        super(SimMouseNet, self).__init__()
        
        self.MouseGraph = Network()
        self.MouseGraph.create_graph()
        
        self.Areas = nn.ModuleDict()
        
        self.hyperparams = yaml.load(open('./SimMouseNet_hyperparams.yaml'), Loader=yaml.FullLoader)
    
    def make_SimMouseNet(self):
        
#         self.Aeas['Retina'] = Retina()
#         self.Areas['LGN'] = LGN()
        
        self.AREAS_LIST = self.MouseGraph.G.nodes
        
        for area in self.AREAS_LIST:
            hyperparams = self.hyperparams['model'][area]
            print(area)
            if area == 'Retina':
                
                self.Areas[area] = Retina(in_channels = hyperparams['in_channels'], 
                                          out_channels = hyperparams['out_channels'], 
                                          kernel_size = hyperparams['kernel_size'], 
                                          padding = hyperparams['padding'])
            
            elif area == 'LGN':
                
                predec_area = list(self.MouseGraph.G.predecessors(area))
                
                this_area_in_channels = self.hyperparams['model'][predec_area[0]]['out_channels']
                
                self.Areas[area] = LGN(in_channels = this_area_in_channels, 
                                       out_channels = hyperparams['out_channels'], 
                                       kernel_size = hyperparams['kernel_size'], 
                                       padding = hyperparams['padding'])
            

            else:
                predec_area = list(self.MouseGraph.G.predecessors(area))
                
                if area == 'VISp':
                
                    this_area_in_channels = self.hyperparams['model'][predec_area[0]]['out_channels']
                
                else:
                    this_area_in_channels = self.hyperparams['model'][predec_area[0]]['L4_out_channels']
                    
                    
                self.Areas[area] = Area(L4_in_channels = this_area_in_channels, 
                                         L4_out_channels = hyperparams['L4_out_channels'], 
                                         L4_kernel_size = hyperparams['L4_kernel_size'], 
                                         L4_padding = hyperparams['L4_padding'],
                                         L2_3_kernel_size = hyperparams['L2_3_kernel_size'], 
                                         L2_3_stride = hyperparams['L2_3_stride'], 
                                         L5_kernel_size = hyperparams['L5_kernel_size'], 
                                         L5_hidden_size = hyperparams['L5_hidden_size'])
                
                    
                

    def forward(self, input):
        
        B, N, C, W, H = input.shape
        input_tempflat = input.view((B*N, C, W, H)).contiguous()

        Out = dict()
        Agg_in = torch.empty((B,N,20,W//4,H//4)).to('cuda')
        for area in self.AREAS_LIST:
            print(area)
            if area == 'Retina':
                Out[area] = self.Areas[area](input_tempflat)
                
            elif area == 'LGN':
                
                Out[area] = self.Areas[area](Out['Retina'])
                
            else:
                predec_area = list(self.MouseGraph.G.predecessors(area))
                
                if area == 'VISp':
                    Out[area] = self.Areas[area](Out[predec_area[0]])
                    B, N, C_, W_, H_ = Out[area][0].shape
                    print(B, N, C_, W_, H_)
                    Out_to_agg = torch.nn.functional.avg_pool2d(Out[area][0].view((B*N, C_, W_, H_)), kernel_size=2, stride=2).contiguous().view((B,N,C_, W_//2, H_//2)).contiguous()
                else:
                    print(Out[predec_area[0]][1].shape)
                    Out[area] = self.Areas[area](Out[predec_area[0]][1])
                    Out_to_agg = Out[area][0]
                    
                
                
                Agg_in = torch.cat((Agg_in,Out_to_agg),dim=2)
            
        return Out, Agg_in
                
                



class Area(nn.Module):
    def __init__(self, 
                    L4_in_channels, L4_out_channels, L4_kernel_size, L4_padding,
                    L2_3_kernel_size, L2_3_stride, 
                    L5_kernel_size, L5_hidden_size,
                ):
        super(Area, self).__init__()

        self.L4 = Layer_4(in_channels = L4_in_channels, out_channels = L4_out_channels)
        self.L2_3 = Layer_2_3(kernel_size = L2_3_kernel_size, stride = L2_3_stride)
        self.L5 = Layer_5(input_size = 2*L4_out_channels, kernel_size = L5_kernel_size, hidden_size = L5_hidden_size)


    def forward(self, input):
        
        
        out_l4 = self.L4(input)
        out_l2_3 = self.L2_3(out_l4)
        
        # to concatenate l4 and l2_3 output to feed to l5
        out_l4_to_l5 = nn.functional.avg_pool2d(out_l4, kernel_size=2, stride=2)
        
        BN_l45, C_l45, W_l45, H_l45 = out_l4_to_l5.shape
        out_l4_to_l5 = out_l4_to_l5.view((B,N, C_l45, W_l45, H_l45)).contiguous()

        BN_l2_3, C_l2_3, W_l2_3, H_l2_3 = out_l2_3.shape
        out_l2_3_to_l5 = out_l2_3.view((B, N, C_l2_3, W_l2_3, H_l2_3)).contiguous()
        
        in_l5 = torch.cat((out_l4_to_l5,out_l2_3_to_l5),dim=2)
        print(in_l5.shape)
        out_l5 = self.L5(in_l5)
        
        return out_l5, out_l2_3

        

class LGN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1):
        super(LGN, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding)
        self.nonlnr = nn.ReLU()
    
    def forward(self, input):

        lgn_out = self.conv(input)
        lgn_out = self.nonlnr(lgn_out)

        return lgn_out

class Retina(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1):
        super(Retina, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding)
        self.nonlnr = nn.ReLU()

    
    def forward(self, input):

        retina_out = self.conv(input)
        retina_out = self.nonlnr(retina_out)

        return retina_out


class Layer_4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1):
        super(Layer_4, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding)
        self.nonlnr = nn.ReLU()

    def forward(self, input):

        l4_out = self.conv(input)
        l4_out = self.nonlnr(l4_out)

        return l4_out

class Layer_2_3(nn.Module):
    def __init__(self, kernel_size = 2, stride = 2):
        super(Layer_2_3, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size = kernel_size, stride = stride)

    def forward(self, input):

        l2_3_out = self.maxpool(input)
        
        return l2_3_out

class Layer_5(nn.Module):
    def __init__(self, input_size, kernel_size, hidden_size = 20):
        super(Layer_5, self).__init__()
        
        self.convgru = ConvGRU(input_size = input_size, hidden_size = hidden_size, kernel_size=1, num_layers=1)
        self.nonlnr = nn.ReLU()

    def forward(self, input):
        
        l5_out, _ = self.convgru(input)
        l5_out = self.nonlnr(l5_out)
        
        return l5_out
        
        
if __name__ == '__main__':
#     retina = Retina(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1).to('cuda')
#     lgn = LGN(in_channels = 32, out_channels = 5, kernel_size = 3, padding = 1).to('cuda')
#     l4 = Layer_4(in_channels = 5, out_channels = 32, kernel_size = 3, padding = 1).to('cuda')
#     l2_3 = Layer_2_3(kernel_size = 2, stride = 2).to('cuda')
#     l5 = Layer_5(input_size = 32, hidden_size=20, kernel_size=1).to('cuda')
#     area = Area(L4_in_channels = 5, L4_out_channels = 32, L4_kernel_size = 3, L4_padding = 1,
#                     L2_3_kernel_size = 2, L2_3_stride = 2, 
#                     L5_input_size = 64, L5_kernel_size = 1, L5_hidden_size = 20).to('cuda')
    sim_mouse_net = SimMouseNet()
    sim_mouse_net.make_SimMouseNet()
    sim_mouse_net.to('cuda')
    
    mydata = torch.FloatTensor(10, 3, 5, 128, 128).to('cuda')
#     mydata = torch.FloatTensor(10, 3, 128, 128)
#     mydata = mydata.permute(0,2,1,3,4).contiguous().view((64,3,128,128))
    nn.init.normal_(mydata)
    import ipdb
    mydata = mydata.permute((0,2,1,3,4)).contiguous()
    B, N, C, W, H = mydata.shape
#     print(B, N, C, W, H)
    out1, out2 = sim_mouse_net(mydata)

#     mydata_tempflat = mydata.view((B*N, C, W, H)).contiguous()
#     out_retina = retina(mydata_tempflat)
#     out_lgn = lgn(out_retina) 
#     out_area1, out_area2 = area(out_lgn)
#     out_l4 = l4(out_lgn)
#     out_l2_3 = l2_3(out_l4)
    
#     out_l4_to_l5 = nn.functional.avg_pool2d(out_l4, kernel_size=2, stride=2)
#     BN_l4, C_l4, W_l4, H_l4 = out_l4_to_l5.shape
#     out_l4_to_l5 = out_l4_to_l5.view((B,N, C_l4, W_l4, H_l4)).contiguous()
#     print(out_l4_to_l5.shape)
    
#     BN_l2_3, C_l2_3, W_l2_3, H_l2_3 = out_l2_3.shape
#     out_l2_3 = out_l2_3.view((B, N, C_l2_3, W_l2_3, H_l2_3)).contiguous()
#     print(out_l2_3.shape)
    
#     out_l5 = l5(out_l2_3)
    
    ipdb.set_trace()