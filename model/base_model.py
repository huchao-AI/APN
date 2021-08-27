import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from .prototype import *
from .layers import *
import pdb
from functools import partial
from .ccattention import ExactNet 

class convAE(torch.nn.Module):
    def __init__(self, n_channel=3,  t_length=5, proto_size=10, feature_dim=512, key_dim=512, temp_update=0.1, temp_gather=0.1):
        super(convAE, self).__init__()

        def Outhead(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )


        self.extract = ExactNet(t_length, n_channel)
        self.prototype = Prototype(proto_size, feature_dim, key_dim, temp_update, temp_gather)
        # output_head
        self.ohead = Outhead(128,n_channel,64)

       
    def set_learnable_params(self, layers):
        for k,p in self.named_parameters():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.named_parameters():
            if p.requires_grad:
                # print(k)
                params[k] = p
        return params

    def get_params(self, layers):
        params = OrderedDict()
        for k, p in self.named_parameters():
            if any([k.startswith(l) for l in layers]):
                # print(k)
                params[k] = p
        return params

    def forward(self, x, weights=None, train=True):
        new_fea = self.extract(x)
        new_fea = F.normalize(new_fea, dim=1)
        
        if train:
            updated_fea, keys, fea_loss, cst_loss, dis_loss, sim_loss = self.prototype(new_fea, new_fea, weights, train)
            if weights == None:
                output = self.ohead(updated_fea)
            else:
                x = conv2d(updated_fea, weights['ohead.0.weight'], weights['ohead.0.bias'], stride=1, padding=1)
                x = relu(x)
                x = conv2d(x, weights['ohead.2.weight'], weights['ohead.2.bias'], stride=1, padding=1)
                x = relu(x)
                x = conv2d(x, weights['ohead.4.weight'], weights['ohead.4.bias'], stride=1, padding=1)
                output = F.tanh(x)

            return output, None, updated_fea, keys, fea_loss, cst_loss, dis_loss, sim_loss
        
        #test
        else:
            updated_fea, keys, query, fea_loss = self.prototype(new_fea, new_fea, weights, train)
            if weights == None:
                output = self.ohead(updated_fea)
            else:
                x = conv2d(updated_fea, weights['ohead.0.weight'], weights['ohead.0.bias'], stride=1, padding=1)
                x = relu(x)
                x = conv2d(x, weights['ohead.2.weight'], weights['ohead.2.bias'], stride=1, padding=1)
                x = relu(x)
                x = conv2d(x, weights['ohead.4.weight'], weights['ohead.4.bias'], stride=1, padding=1)
                output = F.tanh(x)
            
            return output, fea_loss


def dismap(x, name='pred'):
    
    x = x.data.cpu().numpy()
    x = x.mean(1)
    for j in range(x.shape[0]):
        plt.cla()
        y = x[j]
        df = pd.DataFrame(y)
        sns.heatmap(df)
        plt.savefig('results/dismap/{}_{}.png'.format(name,str(j)))
        plt.close()
    return True
