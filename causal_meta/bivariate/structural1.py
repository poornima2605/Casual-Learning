import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
from itertools import chain
from copy import deepcopy

from causal_meta.utils.torch_utils import logsumexp

class BivariateStructuralModel(nn.Module):
    def __init__(self, model_A_B_C, model_C_B_A, model_B_A_C, model_C_A_B, model_A_C_B, model_B_C_A, model_A_B_C_1, model_C_B_A_1, model_B_A_C_1, model_C_A_B_1, model_A_C_B_1, model_B_C_A_1, dtype=torch.float64):
        super(BivariateStructuralModel, self).__init__()
        
        self.model_A_B_C = model_A_B_C
        self.model_C_B_A = model_C_B_A
        self.model_B_A_C = model_B_A_C
        self.model_C_A_B = model_C_A_B
        self.model_A_C_B = model_A_C_B
        self.model_B_C_A = model_B_C_A
        
        self.model_A_B_C_1 = model_A_B_C_1
        self.model_C_B_A_1 = model_C_B_A_1
        self.model_B_A_C_1 = model_B_A_C_1
        self.model_C_A_B_1 = model_C_A_B_1
        self.model_A_C_B_1 = model_A_C_B_1
        self.model_B_C_A_1 = model_B_C_A_1
        
        self.w = nn.Parameter(torch.tensor(0., dtype=dtype))

    def forward(self, inputs):
        return self.online_loglikelihood(self.model_A_B_C(inputs), self.model_C_B_A(inputs), self.model_B_A_C(inputs), self.model_C_A_B(inputs), self.model_A_C_B(inputs), self.model_B_C_A(inputs),self.model_A_B_C_1(inputs), self.model_C_B_A_1(inputs), self.model_B_A_C_1(inputs), self.model_C_A_B_1(inputs), self.model_A_C_B_1(inputs), self.model_B_C_A_1(inputs))

    def online_loglikelihood(self, logl_A_B_C, logl_C_B_A, logl_B_A_C, logl_C_A_B, logl_A_C_B, logl_B_C_A, logl_A_B_C_1, logl_C_B_A_1, logl_B_A_C_1, logl_C_A_B_1, logl_A_C_B_1, logl_B_C_A_1):
        log_alpha, log_1_m_alpha = F.logsigmoid(self.w), F.logsigmoid(-self.w)

        return logsumexp(log_alpha + torch.sum(logl_A_B_C),
            log_1_m_alpha + torch.sum(logl_C_B_A) + log_alpha + torch.sum(logl_C_A_B),
            log_1_m_alpha + torch.sum(logl_A_C_B) + log_alpha + torch.sum(logl_A_C_B),
            log_1_m_alpha + torch.sum(logl_B_C_A)+ torch.sum(logl_A_B_C_1),
            log_1_m_alpha + torch.sum(logl_C_B_A_1) + log_alpha + torch.sum(logl_C_A_B_1),
            log_1_m_alpha + torch.sum(logl_A_C_B_1) + log_alpha + torch.sum(logl_A_C_B_1),
            log_1_m_alpha + torch.sum(logl_B_C_A_1))

    def adapt_modules(self, inputs):
        return -torch.mean(self.model_A_B_C(inputs) + self.model_C_B_A(inputs) + self.model_B_A_C(inputs) + self.model_C_A_B(inputs) + self.model_A_C_B(inputs) + self.model_B_C_A(inputs) +  self.model_A_B_C_1(inputs) + self.model_C_B_A_1(inputs) + self.model_B_A_C_1(inputs) + self.model_C_A_B_1(inputs) + self.model_A_C_B_1(inputs) + self.model_B_C_A_1(inputs))

    def modules_parameters(self):
        return chain(self.model_A_B_C.parameters(), self.model_C_B_A.parameters(), self.model_C_B_A.parameters(), self.model_B_A_C.parameters(), self.model_C_A_B.parameters(), self.model_A_C_B.parameters(), self.model_B_C_A.parameters(), self.model_A_B_C_1.parameters(), self.model_C_B_A_1.parameters(), self.model_C_B_A_1.parameters(), self.model_B_A_C_1.parameters(), self.model_C_A_B_1.parameters(), self.model_A_C_B_1.parameters(), self.model_B_C_A_1.parameters())

    def structural_parameters(self):
        return [self.w]

    @contextlib.contextmanager
    def save_params(self):
        state_dict_A_B_C = deepcopy(self.model_A_B_C.state_dict())
        state_dict_C_B_A = deepcopy(self.model_C_B_A.state_dict())
        state_dict_B_A_C = deepcopy(self.model_A_B_C.state_dict())
        state_dict_C_A_B = deepcopy(self.model_C_B_A.state_dict())
        state_dict_A_C_B = deepcopy(self.model_A_B_C.state_dict())
        state_dict_B_C_A = deepcopy(self.model_C_B_A.state_dict())
        state_dict_A_B_C_1 = deepcopy(self.model_A_B_C_1.state_dict())
        state_dict_C_B_A_1 = deepcopy(self.model_C_B_A_1.state_dict())
        state_dict_B_A_C_1 = deepcopy(self.model_A_B_C_1.state_dict())
        state_dict_C_A_B_1 = deepcopy(self.model_C_B_A_1.state_dict())
        state_dict_A_C_B_1 = deepcopy(self.model_A_B_C_1.state_dict())
        state_dict_B_C_A_1 = deepcopy(self.model_C_B_A_1.state_dict())
        yield
        self.model_A_B_C.load_state_dict(state_dict_A_B_C)
        self.model_C_B_A.load_state_dict(state_dict_C_B_A)
        self.model_B_A_C.load_state_dict(state_dict_B_C_A)
        self.model_C_A_B.load_state_dict(state_dict_C_A_B)
        self.model_A_C_B.load_state_dict(state_dict_A_C_B)
        self.model_B_C_A.load_state_dict(state_dict_B_C_A)
        self.model_A_B_C_1.load_state_dict(state_dict_A_B_C_1)
        self.model_C_B_A_1.load_state_dict(state_dict_C_B_A_1)
        self.model_B_A_C_1.load_state_dict(state_dict_B_C_A_1)
        self.model_C_A_B_1.load_state_dict(state_dict_C_A_B_1)
        self.model_A_C_B_1.load_state_dict(state_dict_A_C_B_1)
        self.model_B_C_A_1.load_state_dict(state_dict_B_C_A_1)

