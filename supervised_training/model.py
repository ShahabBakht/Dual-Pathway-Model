import sys

import torch
import torch.nn as nn

sys.path.append('../backbone/')
sys.path.append("../dpc")
from monkeynet import VisualNet
from model_3d import DPC_RNN


class VisualNet_classifier(nn.Module):
    def __init__(self, num_classes, num_res_blocks = 10, num_paths = 1, pretrained = False, path = './'):
        super().__init__()

        self.visualnet = VisualNet(num_res_blocks = num_res_blocks)
        self.linear = nn.Linear(num_paths * self.visualnet.path1.resblocks_out_channels,num_classes)
        
        if pretrained is True:
            checkpoint = torch.load(path)
            subnet_dict = extract_subnet_dict_from_dpc(checkpoint["state_dict"])
            self.visualnet.load_state_dict(subnet_dict)
        
    def forward(self, x):

        features = self.visualnet(x)
        features = nn.functional.avg_pool3d(features, kernel_size = (features.shape[2], features.shape[3], features.shape[4])).squeeze()
        y = self.linear(features)

        return y
    
class VisualNet_motion(nn.Module):
    def __init__(self, num_classes, num_res_blocks = 10, num_paths = 1, pretrained = False, path = './'):
        super().__init__()

        self.visualnet = VisualNet(num_res_blocks = num_res_blocks)
        self.linear1_1 = nn.Linear(num_paths * self.visualnet.path1.resblocks_out_channels,64)
        self.linear1_2 = nn.Linear(num_paths * self.visualnet.path1.resblocks_out_channels,64)
        self.linear2_1 = nn.Linear(64,2)
        self.linear2_2 = nn.Linear(64,1)
#         self.linear2_1 = nn.Linear(128,1)
#         self.linear2_2 = nn.Linear(128,1)
        self.relu = nn.ReLU()
        
        if pretrained is True:
            checkpoint = torch.load(path)
            subnet_dict = extract_subnet_dict_from_dpc(checkpoint["state_dict"])
            self.visualnet.load_state_dict(subnet_dict)
        
    def forward(self, x):
#         (B, N, C, SL, H, W) = x.shape
#         x = x.view(B*N, C, SL, H, W)
#         N = 3
#         N = x.shape[0]//B
#         x = x.view(B, N, C, SL, H, W).permute(0,2,1,3,4,5).contiguous().view(B,C,N*SL,H,W)
        
        features = self.visualnet(x)
        features = nn.functional.avg_pool3d(features, kernel_size = (features.shape[2], features.shape[3], features.shape[4])).squeeze()
        features_1 = self.linear1_1(features)
        features_2 = self.linear1_2(features)
        features_1 = self.relu(features_1)
        features_2 = self.relu(features_2)
        y_1 = self.linear2_1(features_1)
        y_2 = self.linear2_2(features_2)

#         features_1_aug = torch.cat((features_1,y_2),dim=1)
#         y_1 = self.linear2_1(features_1_aug)

        y = torch.cat((y_1,y_2),1)
#         y = y_1

        return y

class OnePath_classifier(nn.Module):
    def __init__(self, num_classes, num_res_blocks = 10, pretrained = True, path = './', which_path = 'path1'):
        super().__init__()
        

        model = DPC_RNN(
                        sample_size=64, num_seq=8, seq_len=5, network="monkeynet", pred_step=3
                        )
        if pretrained is True:
            checkpoint = torch.load(path)
            subnet_dict = extract_subnet_dict(checkpoint["state_dict"])

            model.load_state_dict(subnet_dict)
        
        self.s1 = model.backbone.s1
        
        if which_path == 'path1':
            self.backbone = model.backbone.path1
        elif which_path == 'path2':
            self.backbone = model.backbone.path2
        
        self.linear = nn.Linear(self.backbone.resblocks_out_channels,num_classes)
    
        
    def forward(self, x):
        
        features = self.s1(x)
        features = self.backbone(features)
        features = nn.functional.avg_pool3d(features, kernel_size = (features.shape[2], features.shape[3], features.shape[4])).squeeze()
        y = self.linear(features)
        
        return y
            
def extract_subnet_dict(d):
        out = {}
        for k, v in d.items():
            if (k.startswith("subnet.") or k.startswith("module.")):
                out[k[7:]] = v
        return out
    
def extract_subnet_dict_from_dpc(d):
        out = {}
        for k, v in d.items():
            if (k.startswith("subnet.") or k.startswith("module.")) and ("backbone" in k):

                out[k[16:]] = v
        return out
    
if __name__ == '__main__':
    
    import time
    
    mydata = torch.FloatTensor(8, 3, 5, 64, 64).to('cuda')
    nn.init.normal_(mydata)
    
    tic = time.time()
#     classifier = VisualNet_classifier(101, num_res_blocks = 10, num_paths = 1).to('cuda')
#     path = '/network/tmp1/bakhtias/Results/log_monkeynet_ucf101_path_2_0/ucf101-64_rnet_dpc-rnn_bs30_lr0.001_seq8_pred3_len5_ds3_train-all/model/epoch100.pth.tar'
#     classifier = OnePath_classifier(10, num_res_blocks = 10, pretrained = True, path = path, which_path = 'path1').to('cuda')
    classifier = VisualNet_motion(num_classes = 2, num_res_blocks = 5, num_paths = 1, pretrained = False, path = './').to('cuda')
#     print(classifier)
    
    agg_out_visual = classifier(mydata)
    
    print(agg_out_visual.shape)

    print(time.time()-tic)
    


