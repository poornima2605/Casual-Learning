import torch
import torch.nn as nn
import numpy as np

class Marginal(nn.Module):
    def __init__(self, N):
        super(Marginal, self).__init__()
        self.N = N
        self.w = nn.Parameter(torch.zeros(N, dtype=torch.float64))
    
    def forward(self, inputs):
        # log p(A) / log p(B)
        cste = torch.logsumexp(self.w, dim=0)
        return self.w[inputs] - cste

    def init_parameters(self):
        self.w.data.zero_()

class Conditional(nn.Module):
    def __init__(self, N):
        super(Conditional, self).__init__()
        self.N = N
        self.w = nn.Parameter(torch.zeros((N, N), dtype=torch.float64))
    
    def forward(self, conds, inputs):
        # log p(B | A) / log p(A | B)
        cste = torch.logsumexp(self.w[conds], dim=1)
        return self.w[conds, inputs] - cste

    def init_parameters(self):
        self.w.data.zero_()

class Conditional_2_variables(nn.Module):
    def __init__(self, N):
        super(Conditional_2_variables, self).__init__()
        self.N = N
        self.w = nn.Parameter(torch.zeros((N, N), dtype=torch.float64))
        self.w1 = nn.Parameter(torch.zeros((N, N), dtype=torch.float64))
    
    def forward(self, conds, inputs1,inputs2):
        # log p(B | A,C) / log p(A | B)
        #inter=np.intersect1d(inputs1,inputs2)
        cste = torch.logsumexp(self.w1[inputs1],dim=1)
        cste1 = torch.logsumexp(self.w1[inputs2],dim=1)
        
        return (self.w[conds, inputs1] - cste) * (self.w[conds, inputs2] - cste1)

    def init_parameters(self):
        self.w.data.zero_()