import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimMouseNet(nn.Module):
    def __init__(self):
        super(SimMouseNet, self).__init__()

        self.Retina = Retina()
        self.LGN = LGN()
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

        pass

class Layer_2_3(nn.Module):
    def __init__(self, kernel_size = 2, stride = 2):
        super(Layer_2_3, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size = kernel_size, stride = stride)

    def forward(self, input):

        pass

class Layer_5(nn.Module):
    def __init__(self):
        super(Layer_5, self).__init__()

    def forward(self, input):

        pass

if __name__ == '__main__':
    mymodel = SimMouseNet(pretrained=False)
    mydata = torch.FloatTensor(10, 3, 5, 128, 128)
#     mydata = mydata.permute(0,2,1,3,4).contiguous().view((64,3,128,128))
    nn.init.normal_(mydata)
    import ipdb; ipdb.set_trace()
    mymodel(mydata)