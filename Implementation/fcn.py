import torch.nn as nn 
import torch.nn.functional as F  

class FCN8s( nn.Module ):
    def __init__(self, n_classes=21):
        super(FCN8s,self).__init__()
        self.n_classes = n_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3,64,3,padding=100)
        )