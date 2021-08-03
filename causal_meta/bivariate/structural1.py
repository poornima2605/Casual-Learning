import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
from itertools import chain
from copy import deepcopy

from causal_meta.utils.torch_utils import logsumexp

class BivariateStructuralModel(nn.Module):
    def __init__(self, model_A_B_C, model_C_B_A, dtype=torch.float64):
        super(BivariateStructuralModel, self).__init__()
        self.model_A_B_C = model_A_B_C
        self.model_C_B_A = model_C_B_A
        self.w = nn.Parameter(torch.tensor(0., dtype=dtype))

    def forward(self, inputs):
        return self.online_loglikelihood(self.model_A_B_C(inputs), self.model_C_B_A(inputs))

    def online_loglikelihood(self, logl_A_B_C, logl_C_B_A):
        log_alpha, log_1_m_alpha = F.logsigmoid(self.w), F.logsigmoid(-self.w)

        return logsumexp(log_alpha + torch.sum(logl_A_B_C),
            log_1_m_alpha + torch.sum(logl_C_B_A))

    def adapt_modules(self, inputs):
        return -torch.mean(self.model_A_B_C(inputs) + self.model_C_B_A(inputs))

    def modules_parameters(self):
        return chain(self.model_A_B_C.parameters(), self.model_C_B_A.parameters())

    def structural_parameters(self):
        return [self.w]

    @contextlib.contextmanager
    def save_params(self):
        state_dict_A_B_C = deepcopy(self.model_A_B_C.state_dict())
        state_dict_C_B_A = deepcopy(self.model_C_B_A.state_dict())
        yield
        self.model_A_B_C.load_state_dict(state_dict_A_B_C)
        self.model_C_B_A.load_state_dict(state_dict_C_B_A)
