#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.nn.functional as F
import numpy as np



def gram_matrix(x):
    n,c,h,w=x.size()
    x=x.view(n*c,h*w)
    gram=torch.mm(x, x.t())
    return gram

class FeatureLoss(nn.Module):
    def __init__(self):
        super(FeatureLoss, self).__init__()        
        self.criterion = nn.MSELoss() 
    def forward(self, pred_features, target_features):
        
        n,c,h,w=pred_features.shape
        pred_gram=gram_matrix(pred_features)
        target_gram=gram_matrix(target_features)
        loss = self.criterion(pred_gram, target_gram)
        return loss
    
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
#        self.weights = [0, 1.0/32, 1.0/16, 1.0/8, 1.0/4]
#        self.weights = [0, 0, 0, 0, 1.0/4]
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0/2]

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        from torchvision import models
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
    
    

if __name__=="__main__":
    test_input=torch.rand(6, 1, 128, 128)
    grad=torch.rand(6, 1, 128, 128)
#    print("input_size:",test_input.size())
    model=KLLoss()
    loss = model(test_input, grad)
    print(loss)
#    map_A = model(test_input)
#    print("output_size:", np.size(map_A))