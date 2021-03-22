import sys

import torch
import torch.nn as nn

sys.path.append('../backbone/')
from monkeynet import VisualNet

class VisualNet_classifier(nn.Module):
    def __init__(self, num_classes, num_res_blocks = 10, num_paths = 1):
        super().__init__()

        self.visualnet = VisualNet(num_res_blocks = num_res_blocks)
        self.linear = nn.Linear(num_paths * self.visualnet.path1.resblocks_out_channels,num_classes)

    def forward(self, x):

        features = self.visualnet(x)
        features = nn.functional.avg_pool3d(features, kernel_size = (features.shape[2], features.shape[3], features.shape[4])).squeeze()
        y = self.linear(features)

        return y
    
if __name__ == '__main__':
    
    import time
    
    mydata = torch.FloatTensor(8, 3, 5, 64, 64).to('cuda')
    nn.init.normal_(mydata)
    
    tic = time.time()
    classifier = VisualNet_classifier(101, num_res_blocks = 10, num_paths = 1).to('cuda')
    print(classifier)
    
    agg_out_visual = classifier(mydata)
    
    print(agg_out_visual.shape)

    print(time.time()-tic)
    


