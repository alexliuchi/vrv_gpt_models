'''
vrv
@author: Alex Liu
@description:
@date:
'''

import torch
import torch.nn as nn

class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
