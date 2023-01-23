from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class GenericFace(ABC, nn.Module):
    """ Base class for ArcFace, SphereFace, CosFace, GAMP loss classes
    """
    def __init__(self, feat_dim, num_class, s, m):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.s = s
        self.m = m
        self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_normal_(self.w)

    @abstractmethod
    def forward(self, x, y):
        pass 
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'

