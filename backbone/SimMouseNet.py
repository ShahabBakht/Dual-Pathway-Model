import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from convrnn import ConvGRU


class SimMouseNet(nn.Module):
    def __init__(self):
        super(SimMouseNet, self).__init__()

        self.Retina = Retina(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.LGN = LGN(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.VISp = Area()
        self.VISal = Area()
        self.VISl = Area()
        self.VISpm = Area()
        self.VISam = Area()

    def forward(self, input):
        pass


class Area(nn.Module):
    def __init__(self, 
                    L4_in_channels, L4_out_channels, L4_kernel_size, L4_padding,
                    L2_3_kernel_size, L2_3_stride, 
                    L5_out_channels, L5_kernel_size, L5_passing,
                ):
        super(Area, self).__init__()

        self.L4 = Layer_4(in_channels = L4_in_channels, out_channels = L4_out_channels)
        self.L2_3 = Layer_2_3(kernel_size = L2_3_kernel_size, stride = L2_3_stride)
        self.L5 = Layer_5()


    def forward(self, input):
        pass


class LGN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1):
        super(LGN, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding)
    
    def forward(self, input):

        lgn_out = self.conv(input)

        return lgn_out

class Retina(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1):
        super(Retina, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding)
    
    def forward(self, input):

        retina_out = self.conv(input)

        return retina_out


class Layer_4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1):
        super(Layer_4, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding)

    def forward(self, input):

        l4_out = self.conv(input)

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
        

    def forward(self, input):
        
        l5_out = self.convgru(input)
        
        return l5_out
        
        
if __name__ == '__main__':
    retina = Retina(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1).to('cuda')
    lgn = LGN(in_channels = 32, out_channels = 5, kernel_size = 3, padding = 1).to('cuda')
    l4 = Layer_4(in_channels = 5, out_channels = 32, kernel_size = 3, padding = 1).to('cuda')
    l2_3 = Layer_2_3(kernel_size = 2, stride = 2).to('cuda')
    l5 = Layer_5(input_size = 32, hidden_size=20, kernel_size=1).to('cuda')
    mydata = torch.FloatTensor(10, 3, 5, 128, 128).to('cuda')
#     mydata = torch.FloatTensor(10, 3, 128, 128)
#     mydata = mydata.permute(0,2,1,3,4).contiguous().view((64,3,128,128))
    nn.init.normal_(mydata)
    import ipdb
    mydata = mydata.permute((0,2,1,3,4)).contiguous()
    B, N, C, W, H = mydata.shape

    mydata_tempflat = mydata.view((B*N, C, W, H)).contiguous()
    out_retina = retina(mydata_tempflat)
    out_lgn = lgn(out_retina) 
    out_l4 = l4(out_lgn)
    out_l2_3 = l2_3(out_l4)
    BN, C, W, H = out_l2_3.shape
    out_l2_3 = out_l2_3.view((B,N,C,W,H)).contiguous()
#     out_l2_3 = out_l2_3.permute((0,2,1,3,4)).contiguous()
    out_l5, last_state_list = l5(out_l2_3)
    
    ipdb.set_trace()