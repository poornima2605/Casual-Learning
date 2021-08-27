import numpy as np

import torch
import torch.nn as nn

from causal_meta.modules.categorical import Marginal, Conditional
from causal_meta.bivariate.structural1 import BivariateStructuralModel

class Model(nn.Module):
    def __init__(self, N):
        super(Model, self).__init__()
        self.N = N

    def set_maximum_likelihood(self, inputs):
        inputs_A, inputs_B , inputs_C = np.split(inputs.numpy(), 3, axis=1)
        num_samples = inputs_A.shape[0]
        
        pi_A = np.zeros((self.N,), dtype=np.float64)
        pi_B_A = np.zeros((self.N, self.N), dtype=np.float64)
        pi_C_B = np.zeros((self.N, self.N), dtype=np.float64)
        
        # Empirical counts for p(A)
        for i in range(num_samples):
            pi_A[inputs_A[i, 0]] += 1
        pi_A /= float(num_samples)
        assert np.isclose(np.sum(pi_A, axis=0), 1.)
        
        # Empirical counts for p(B | A)
        for i in range(num_samples):
            pi_B_A[inputs_A[i, 0], inputs_B[i, 0]] += 1
        pi_B_A /= np.maximum(np.sum(pi_B_A, axis=1, keepdims=True), 1.)
        sum_pi_B_A = np.sum(pi_B_A, axis=1)
        assert np.allclose(sum_pi_B_A[sum_pi_B_A > 0], 1.)
        
        # Empirical counts for p(C | B)
        for i in range(num_samples):
            pi_C_B[inputs_B[i, 0], inputs_C[i, 0]] += 1
        pi_C_B /= np.maximum(np.sum(pi_C_B, axis=1, keepdims=True), 1.)
        sum_pi_C_B = np.sum(pi_C_B, axis=1)
        assert np.allclose(sum_pi_C_B[sum_pi_C_B > 0], 1.)


        return self.set_analytical_maximum_likelihood(pi_A, pi_B_A, pi_C_B)
    

class Model1(Model):
    def __init__(self, N):
        super(Model1, self).__init__(N=N)
        self.p_A = Marginal(N)
        self.p_B_A = Conditional(N)
        self.p_C_B = Conditional(N)
    
    def forward(self, inputs):
        # Compute the (negative) log-likelihood with the
        # decomposition p(x_A, x_B ,x_C) = p(x_A) p(x_B | x_A) p(x_C | x_B)
        #print(inputs)
        inputs_A, inputs_B, inputs_C = torch.split(inputs, 1, dim=1)
        #print(inputs_C)
        inputs_A, inputs_B, inputs_C = inputs_A.squeeze(1), inputs_B.squeeze(1), inputs_C.squeeze(1)

        return self.p_A(inputs_A) + self.p_B_A(inputs_A, inputs_B) + self.p_C_B(inputs_B, inputs_C)

    def set_analytical_maximum_likelihood(self, pi_A, pi_B_A, pi_C_B):
        pi_A_th = torch.from_numpy(pi_A)
        pi_B_A_th = torch.from_numpy(pi_B_A)
        pi_C_B_th = torch.from_numpy(pi_C_B)
        
        self.p_A.w.data = torch.log(pi_A_th)
        self.p_B_A.w.data = torch.log(pi_B_A_th)
        self.p_C_B.w.data = torch.log(pi_C_B_th)

    def init_parameters(self):
        self.p_A.init_parameters()
        self.p_B_A.init_parameters()
        self.p_C_B.init_parameters()
        
class Model2(Model):
    def __init__(self, N):
        super(Model2, self).__init__(N=N)
        self.p_C = Marginal(N)
        self.p_B_C = Conditional(N)
        self.p_A_B = Conditional(N)

    def forward(self, inputs):
        # Compute the (negative) log-likelihood with the
        # decomposition p(x_A, x_B, x_C) = p(x_C)p(x_B | x_C)p(x_A|x_B)
        
        inputs_A, inputs_B, inputs_C = torch.split(inputs, 1, dim=1)
        inputs_A, inputs_B, inputs_C = inputs_A.squeeze(1), inputs_B.squeeze(1), inputs_C.squeeze(1)
        
        return self.p_C(inputs_C) + self.p_B_C(inputs_C, inputs_B) + self.p_A_B(inputs_B, inputs_A) 

    def set_analytical_maximum_likelihood(self, pi_A, pi_B_A, pi_C_B):
        pi_A_th = torch.from_numpy(pi_A)
        pi_B_A_th = torch.from_numpy(pi_B_A)
        pi_C_B_th = torch.from_numpy(pi_C_B)
        
        log_joint1 = torch.log(pi_A_th.unsqueeze(1)) + torch.log(pi_B_A_th)
        log_p_B = torch.logsumexp(log_joint1, dim=0)
        
        log_joint2 = torch.log(log_p_B.unsqueeze(1)) + torch.log(pi_C_B_th)
        log_p_C = torch.logsumexp(log_joint2, dim=0)
        
        self.p_C.w.data = log_p_C
        self.p_B_C.w.data = log_joint2.t() - log_p_C.unsqueeze(1)
        self.p_A_B.w.data = log_joint1.t() - log_p_B.unsqueeze(1)

    def init_parameters(self):
        self.p_C.init_parameters()
        self.p_B_C.init_parameters()
        self.p_A_B.init_parameters()
        
class Model3(Model):
    def __init__(self, N):
        super(Model3, self).__init__(N=N)
        self.p_B = Marginal(N)
        self.p_A_B = Conditional(N)
        self.p_C_A = Conditional(N)

    def forward(self, inputs):
        # Compute the (negative) log-likelihood with the
        # decomposition p(x_A, x_B, x_C) = p(x_B)p(x_A | x_B)p(x_C|x_A)
        
        inputs_A, inputs_B, inputs_C = torch.split(inputs, 1, dim=1)
        inputs_A, inputs_B, inputs_C = inputs_A.squeeze(1), inputs_B.squeeze(1), inputs_C.squeeze(1)
        
        return self.p_B(inputs_B) + self.p_A_B(inputs_B, inputs_A) + self.p_C_A(inputs_A, inputs_C) 

    def set_analytical_maximum_likelihood(self, pi_A, pi_B_A, pi_C_B):
        pi_A_th = torch.from_numpy(pi_A)
        pi_B_A_th = torch.from_numpy(pi_B_A)
        pi_C_B_th = torch.from_numpy(pi_C_B)
        
        log_joint1 = torch.log(pi_A_th.unsqueeze(1)) + torch.log(pi_B_A_th)
        log_p_B = torch.logsumexp(log_joint1, dim=0)
        
        log_joint2 = torch.log(log_p_B.unsqueeze(1)) + torch.log(pi_C_B_th) 
        log_p_C = torch.logsumexp(log_joint2, dim=0)
        
        log_joint3 = log_p_C + torch.log(pi_C_B_th) + torch.log(pi_B_A_th)
        
        self.p_B.w.data = log_p_B
        self.p_A_B.w.data = log_joint1.t() - log_p_B.unsqueeze(1)
        self.p_C_A.w.data = log_joint3.t() - torch.log(pi_A_th.unsqueeze(1))

    def init_parameters(self):
        self.p_B.init_parameters()
        self.p_A_B.init_parameters()
        self.p_C_A.init_parameters()
        
class Model4(Model):
    def __init__(self, N):
        super(Model4, self).__init__(N=N)
        self.p_C = Marginal(N)
        self.p_A_C = Conditional(N)
        self.p_B_A = Conditional(N)

    def forward(self, inputs):
        # Compute the (negative) log-likelihood with the
        # decomposition p(x_A, x_B, x_C) = p(x_C)p(x_A | x_C)p(x_B|x_A)
        
        inputs_A, inputs_B, inputs_C = torch.split(inputs, 1, dim=1)
        inputs_A, inputs_B, inputs_C = inputs_A.squeeze(1), inputs_B.squeeze(1), inputs_C.squeeze(1)
        
        return self.p_C(inputs_C) + self.p_A_C(inputs_C, inputs_A) + self.p_B_A(inputs_A, inputs_B) 

    def set_analytical_maximum_likelihood(self, pi_A, pi_B_A, pi_C_B):
        pi_A_th = torch.from_numpy(pi_A)
        pi_B_A_th = torch.from_numpy(pi_B_A)
        pi_C_B_th = torch.from_numpy(pi_C_B)
        
        log_joint1 = torch.log(pi_A_th.unsqueeze(1)) + torch.log(pi_B_A_th)
        log_p_B = torch.logsumexp(log_joint1, dim=0)
        
        log_joint2 = torch.log(log_p_B.unsqueeze(1)) + torch.log(pi_C_B_th) 
        log_p_C = torch.logsumexp(log_joint2, dim=0)
        
        log_p_C_A = (log_p_C + torch.log(pi_C_B_th) + torch.log(pi_B_A_th)).t - torch.log(pi_A_th.unsqueeze(1))
        log_joint3 = torch.log(pi_A_th.unsqueeze(1)) + log_p_C_A
        
        self.p_C.w.data = log_p_C
        self.p_A_C.w.data = log_joint3.t() - log_p_C.unsqueeze(1)
        self.p_B_A.w.data = torch.log(pi_B_A_th)

    def init_parameters(self):
        self.p_C.init_parameters()
        self.p_A_C.init_parameters()
        self.p_B_A.init_parameters()
        
class Model5(Model):
    def __init__(self, N):
        super(Model5, self).__init__(N=N)
        self.p_A = Marginal(N)
        self.p_C_A = Conditional(N)
        self.p_B_C = Conditional(N)

    def forward(self, inputs):
        # Compute the (negative) log-likelihood with the
        # decomposition p(x_A, x_B, x_C) = p(x_C)p(x_A | x_C)p(x_B|x_A)
        
        inputs_A, inputs_B, inputs_C = torch.split(inputs, 1, dim=1)
        inputs_A, inputs_B, inputs_C = inputs_A.squeeze(1), inputs_B.squeeze(1), inputs_C.squeeze(1)
        
        return self.p_A(inputs_A) + self.p_C_A(inputs_A, inputs_C) + self.p_B_C(inputs_C, inputs_B) 

    def set_analytical_maximum_likelihood(self, pi_A, pi_B_A, pi_C_B):
        pi_A_th = torch.from_numpy(pi_A)
        pi_B_A_th = torch.from_numpy(pi_B_A)
        pi_C_B_th = torch.from_numpy(pi_C_B)
        
        log_joint1 = torch.log(pi_A_th.unsqueeze(1)) + torch.log(pi_B_A_th)
        log_p_B = torch.logsumexp(log_joint1, dim=0)
        
        log_joint2 = torch.log(log_p_B.unsqueeze(1)) + torch.log(pi_C_B_th) 
        log_p_C = torch.logsumexp(log_joint2, dim=0)
        
        log_joint3 = log_p_C + torch.log(pi_C_B_th) + torch.log(pi_B_A_th) 
        
        self.p_A.w.data = torch.log(pi_A_th)
        self.p_C_A.w.data = log_joint3.t() - torch.log(pi_A_th.unsqueeze(1))
        self.p_B_C.w.data = log_joint2.t() - log_p_C.unsqueeze(1)

    def init_parameters(self):
        self.p_A.init_parameters()
        self.p_C_A.init_parameters()
        self.p_B_C.init_parameters()
        
class Model6(Model):
    def __init__(self, N):
        super(Model6, self).__init__(N=N)
        self.p_B = Marginal(N)
        self.p_C_B = Conditional(N)
        self.p_A_C = Conditional(N)

    def forward(self, inputs):
        # Compute the (negative) log-likelihood with the
        # decomposition p(x_A, x_B, x_C) = p(x_C)p(x_A | x_C)p(x_B|x_A)
        
        inputs_A, inputs_B, inputs_C = torch.split(inputs, 1, dim=1)
        inputs_A, inputs_B, inputs_C = inputs_A.squeeze(1), inputs_B.squeeze(1), inputs_C.squeeze(1)
        
        return self.p_B(inputs_B) + self.p_C_B(inputs_B, inputs_C) + self.p_A_C(inputs_C, inputs_A) 

    def set_analytical_maximum_likelihood(self, pi_A, pi_B_A, pi_C_B):
        pi_A_th = torch.from_numpy(pi_A)
        pi_B_A_th = torch.from_numpy(pi_B_A)
        pi_C_B_th = torch.from_numpy(pi_C_B)
        
        log_joint1 = torch.log(pi_A_th.unsqueeze(1)) + torch.log(pi_B_A_th)
        log_p_B = torch.logsumexp(log_joint1, dim=0)
        
        log_joint2 = torch.log(log_p_B.unsqueeze(1)) + torch.log(pi_C_B_th) 
        log_p_C = torch.logsumexp(log_joint2, dim=0)
        
        log_p_C_A = (log_p_C + torch.log(pi_C_B_th) + torch.log(pi_B_A_th)).t - torch.log(pi_A_th.unsqueeze(1))
        log_joint3 = torch.log(pi_A_th.unsqueeze(1)) + log_p_C_A
        
        self.p_B.w.data = log_p_B
        self.p_C_B.w.data = torch.log(pi_C_B_th)
        self.p_A_C.w.data = log_joint3.t() - log_p_C.unsqueeze(1)

    def init_parameters(self):
        self.p_B.init_parameters()
        self.p_C_B.init_parameters()
        self.p_A_C.init_parameters()
                
class Model7(Model):
    def __init__(self, N):
        super(Model7, self).__init__(N=N)
        self.p_B = Marginal(N)
        self.p_A_B = Conditional(N)
        self.p_C_B = Conditional(N)
    
    def forward(self, inputs):
        # Compute the (negative) log-likelihood with the
        # decomposition p(x_A, x_B ,x_C) = p(x_B) p(x_A | x_B) p(x_C | x_B)
        
        inputs_A, inputs_B, inputs_C = torch.split(inputs, 1, dim=1)
        
        inputs_A, inputs_B, inputs_C = inputs_A.squeeze(1), inputs_B.squeeze(1), inputs_C.squeeze(1)

<<<<<<< HEAD
        return self.p_B(inputs_B) + self.p_A_B(inputs_B, inputs_A) + self.p_C_B(inputs_B, inputs_C)
=======
        return self.p_B(inputs_B) + self.p_B_A(inputs_A, inputs_B) + self.p_B_C(inputs_C, inputs_B)
>>>>>>> 5ab83fc79edfe133de75bb23df034bace1164911

    def set_analytical_maximum_likelihood(self, pi_A, pi_B_A,pi_C_B):
        pi_A_th = torch.from_numpy(pi_A)
        pi_B_A_th = torch.from_numpy(pi_B_A)
        pi_C_B_th = torch.from_numpy(pi_B_C)
        
        log_joint1 = torch.log(pi_A_th.unsqueeze(1)) + torch.log(pi_B_A_th)
        log_p_B = torch.logsumexp(log_joint1, dim=0)
        
        log_joint2 = torch.log(log_p_B.unsqueeze(1)) + torch.log(pi_C_B_th) 
        log_p_C = torch.logsumexp(log_joint2, dim=0)
        
        self.p_B.w.data = log_p_B
        self.p_A_B.w.data = log_joint1.t() - log_p_B.unsqueeze(1)
        self.p_C_B.w.data = torch.log(pi_C_B_th)

    def init_parameters(self):
        self.p_B.init_parameters()
        self.p_A_B.init_parameters()
        self.p_C_B.init_parameters()
        
class Model8(Model):
    def __init__(self, N):
        super(Model8, self).__init__(N=N)
        self.p_A = Marginal(N)
        self.p_B_A = Conditional(N)
        self.p_B_C = Conditional(N)
        self.p_C = Marginal(N)
        self.p_B = Marginal(N)

    def forward(self, inputs):
        # Compute the (negative) log-likelihood with the
<<<<<<< HEAD
        # decomposition p(x_A, x_B, x_C) = p(x_C)p(x_B | x_A, x_C)p(x_A)
=======
        # decomposition p(x_A, x_B, x_C) = p(x_C)p(x_B | x_C)p(x_A | x_B)
>>>>>>> 5ab83fc79edfe133de75bb23df034bace1164911
        
        inputs_A, inputs_B, inputs_C = torch.split(inputs, 1, dim=1)
        inputs_A, inputs_B, inputs_C = inputs_A.squeeze(1), inputs_B.squeeze(1), inputs_C.squeeze(1)
        
        return self.p_A(inputs_A) + (self.p_B_A(inputs_A, inputs_B) * self.p_B_C(inputs_C, inputs_B) * self.p_B(inputs_B))/ (self.p_A(inputs_A) * self.p_C(inputs_C)) + self.p_C(inputs_C) 

    def set_analytical_maximum_likelihood(self, pi_A, pi_B_A, pi_C_B):
        pi_A_th = torch.from_numpy(pi_A)
        pi_B_A_th = torch.from_numpy(pi_B_A)
        pi_C_B_th = torch.from_numpy(pi_B_C)
        
        log_joint1 = torch.log(pi_A_th.unsqueeze(1)) + torch.log(pi_B_A_th)
        log_p_A = torch.logsumexp(log_joint1, dim=0)
        
        log_joint2 = torch.log(log_p_B.unsqueeze(1)) + torch.log(pi_C_B_th) 
        log_p_C = torch.logsumexp(log_joint2, dim=0)
        
        self.p_A.w.data = log_p_A
        self.p_B_A.w.data = torch.log(pi_B_A_th)
        self.p_B_C.w.data = log_joint2.t() - log_p_C.unsqueeze(1)
        self.p_C.w.data = log_p_C

    def init_parameters(self):
        self.p_B.init_parameters()
        self.p_B_A.init_parameters()
        self.p_B_C.init_parameters()
        self.p_C.init_parameters()
        self.p_A.init_parameters()
        
class Model9(Model):
    def __init__(self, N):
        super(Model9, self).__init__(N=N)
        self.p_A = Marginal(N)
        self.p_B_A = Conditional(N)
        self.p_C_A = Conditional(N)
    
    def forward(self, inputs):
        # Compute the (negative) log-likelihood with the
        # decomposition p(x_A, x_B ,x_C) = p(x_A) p(x_B | x_A) p(x_C | x_A)
        
        inputs_A, inputs_B, inputs_C = torch.split(inputs, 1, dim=1)
        
        inputs_A, inputs_B, inputs_C = inputs_A.squeeze(1), inputs_B.squeeze(1), inputs_C.squeeze(1)

        return self.p_A(inputs_A) + self.p_B_A(inputs_A, inputs_B) + self.p_C_A(inputs_A, inputs_C)

    def set_analytical_maximum_likelihood(self, pi_A, pi_B_A, pi_C_B):
        pi_A_th = torch.from_numpy(pi_A)
        pi_B_A_th = torch.from_numpy(pi_B_A)
        pi_C_B_th = torch.from_numpy(pi_B_C)
        
        log_joint1 = torch.log(pi_A_th.unsqueeze(1)) + torch.log(pi_B_A_th)
        log_p_B = torch.logsumexp(log_joint1, dim=0)
        
        log_joint2 = torch.log(log_p_B.unsqueeze(1)) + torch.log(pi_C_B_th) 
        log_p_C = torch.logsumexp(log_joint2, dim=0)
        
        log_joint3 = log_p_C + torch.log(pi_C_B_th) + torch.log(pi_B_A_th) 
        
        self.p_A.w.data = torch.log(pi_A_th)
        self.p_B_A.w.data = torch.log(pi_B_A_th)
        self.p_C_A.w.data = log_joint3.t() - torch.log(pi_A_th.unsqueeze(1))

    def init_parameters(self):
        self.p_B.init_parameters()
        self.p_B_A.init_parameters()
        self.p_C_A.init_parameters()
        
class Model10(Model):
    def __init__(self, N):
        super(Model10, self).__init__(N=N)
        self.p_A = Marginal(N)
        self.p_A_B = Conditional(N)
        self.p_A_C = Conditional(N)
        self.p_C = Marginal(N)
        self.p_B = Marginal(N)

    def forward(self, inputs):
        # Compute the (negative) log-likelihood with the
        # decomposition p(x_A, x_B, x_C) = p(x_C)p(x_B | x_A, x_C)p(x_A)
        
        inputs_A, inputs_B, inputs_C = torch.split(inputs, 1, dim=1)
        inputs_A, inputs_B, inputs_C = inputs_A.squeeze(1), inputs_B.squeeze(1), inputs_C.squeeze(1)
        
        return self.p_A(inputs_A) + (self.p_A_B(inputs_B, inputs_A) * self.p_A_C(inputs_C, inputs_A) * self.p_A(inputs_A))/ (self.p_B(inputs_B) * self.p_C(inputs_C)) + self.p_C(inputs_C) 

    def set_analytical_maximum_likelihood(self, pi_A, pi_B_A, pi_C_B):
        pi_B_th = torch.from_numpy(pi_B)
        pi_B_A_th = torch.from_numpy(pi_B_A)
        pi_C_B_th = torch.from_numpy(pi_B_C)
        
        log_joint1 = torch.log(pi_A_th.unsqueeze(1)) + torch.log(pi_B_A_th)
        log_p_A = torch.logsumexp(log_joint1, dim=0)
        
        log_joint2 = torch.log(log_p_B.unsqueeze(1)) + torch.log(pi_C_B_th) 
        log_p_C = torch.logsumexp(log_joint2, dim=0)
        
        log_p_C_A = (log_p_C + torch.log(pi_C_B_th) + torch.log(pi_B_A_th)).t - torch.log(pi_A_th.unsqueeze(1))
        log_joint3 = torch.log(pi_A_th.unsqueeze(1)) + log_p_C_A
        
        self.p_A.w.data = log_p_A
        self.p_A_B.w.data = log_joint1.t() - log_p_B.unsqueeze(1)
        self.p_A_C.w.data = log_joint3.t() - log_p_C.unsqueeze(1)
        self.p_C.w.data = log_p_C

    def init_parameters(self):
        self.p_B.init_parameters()
        self.p_A_B.init_parameters()
        self.p_A_C.init_parameters()
        self.p_C.init_parameters()
        self.p_A.init_parameters()
        
class Model11(Model):
    def __init__(self, N):
        super(Model11, self).__init__(N=N)
        self.p_C = Marginal(N)
        self.p_A_C = Conditional(N)
        self.p_B_C = Conditional(N)
    
    def forward(self, inputs):
        # Compute the (negative) log-likelihood with the
        # decomposition p(x_A, x_B ,x_C) = p(x_C) p(x_A | x_C) p(x_B | x_C)
        
        inputs_A, inputs_B, inputs_C = torch.split(inputs, 1, dim=1)
        
        inputs_A, inputs_B, inputs_C = inputs_A.squeeze(1), inputs_B.squeeze(1), inputs_C.squeeze(1)

        return self.p_C(inputs_C) + self.p_A_C(inputs_C, inputs_A) + self.p_B_C(inputs_C, inputs_B)

    def set_analytical_maximum_likelihood(self, pi_A, pi_B_A, pi_C_B):
        pi_A_th = torch.from_numpy(pi_A)
        pi_B_A_th = torch.from_numpy(pi_B_A)
        pi_C_B_th = torch.from_numpy(pi_B_C)
        
        log_joint1 = torch.log(pi_A_th.unsqueeze(1)) + torch.log(pi_B_A_th)
        log_p_B = torch.logsumexp(log_joint1, dim=0)
        
        log_joint2 = torch.log(log_p_B.unsqueeze(1)) + torch.log(pi_C_B_th) 
        log_p_C = torch.logsumexp(log_joint2, dim=0)
         
        log_p_C_A = (log_p_C + torch.log(pi_C_B_th) + torch.log(pi_B_A_th)).t - torch.log(pi_A_th.unsqueeze(1))
        log_joint3 = torch.log(pi_A_th.unsqueeze(1)) + log_p_C_A
        
        self.p_C.w.data = torch.log(pi_A_th)
        self.p_A_C.w.data =  log_joint3.t() - log_p_C.unsqueeze(1)
        self.p_B_C.w.data = log_joint2.t() - log_p_C.unsqueeze(1)

    def init_parameters(self):
        self.p_C.init_parameters()
        self.p_A_C.init_parameters()
        self.p_B_C.init_parameters()
        
class Model12(Model):
    def __init__(self, N):
        super(Model12, self).__init__(N=N)
        self.p_A = Marginal(N)
        self.p_C_A = Conditional(N)
        self.p_C_B = Conditional(N)
        self.p_C = Marginal(N)
        self.p_B = Marginal(N)

    def forward(self, inputs):
        # Compute the (negative) log-likelihood with the
<<<<<<< HEAD
        # decomposition p(x_A, x_B, x_C) = p(x_A)p(x_C | x_A, x_B)p(x_B)
=======
        # decomposition p(x_A, x_B, x_C) = p(x_C)p(x_B | x_A, x_C)p(x_A)
        # Naive Bayes Theorem p(x_B | x_A, x_C)= ( p(x_B) p(x_A | x_B) p(x_C | x_B) ) / ( p(x_A) p(x_B))
>>>>>>> 5ab83fc79edfe133de75bb23df034bace1164911
        
        inputs_A, inputs_B, inputs_C = torch.split(inputs, 1, dim=1)
        inputs_A, inputs_B, inputs_C = inputs_A.squeeze(1), inputs_B.squeeze(1), inputs_C.squeeze(1)
        
<<<<<<< HEAD
        return self.p_A(inputs_A) + (self.p_C_A(inputs_A, inputs_C) * self.p_C_B(inputs_B, inputs_C) * self.p_C(inputs_C))/ (self.p_B(inputs_B) * self.p_A(inputs_A)) + self.p_B(inputs_B) 
=======
        return self.p_A(inputs_A) + (self.p_A_B(inputs_A, inputs_B) * self.p_C_B(inputs_C, inputs_B) * self.p_B(inputs_B))/ (self.p_A(inputs_A) * self.p_C(inputs_C)) + self.p_C(inputs_C) 
>>>>>>> 5ab83fc79edfe133de75bb23df034bace1164911

    def set_analytical_maximum_likelihood(self, pi_A, pi_B_A, pi_C_B):
        pi_B_th = torch.from_numpy(pi_B)
        pi_B_A_th = torch.from_numpy(pi_B_A)
        pi_C_B_th = torch.from_numpy(pi_B_C)
        
        log_joint1 = torch.log(pi_A_th.unsqueeze(1)) + torch.log(pi_B_A_th)
        log_p_A = torch.logsumexp(log_joint1, dim=0)
        
        log_joint2 = torch.log(log_p_B.unsqueeze(1)) + torch.log(pi_C_B_th) 
        log_p_C = torch.logsumexp(log_joint2, dim=0)
        
        log_joint3 = log_p_C + torch.log(pi_C_B_th) + torch.log(pi_B_A_th) 
        
        self.p_A.w.data = log_p_A
        self.p_C_A.w.data = log_joint3.t() - torch.log(pi_A_th.unsqueeze(1))
        self.p_C_B.w.data = torch.log(pi_C_B_th)
        self.p_C.w.data = log_p_C

    def init_parameters(self):
        self.p_B.init_parameters()
        self.p_C_A.init_parameters()
        self.p_C_B.init_parameters()
        self.p_C.init_parameters()
        self.p_A.init_parameters()
        
class StructuralModel(BivariateStructuralModel):
    def __init__(self, num_categories):
        
        model_A_B_C = Model1(num_categories)
        model_C_B_A = Model2(num_categories)
        
        model_B_A_C = Model3(num_categories)
        model_C_A_B = Model4(num_categories)
        
        model_A_C_B = Model5(num_categories)
        model_B_C_A = Model6(num_categories)
        
        model_A_B_C_1 = Model7(num_categories)
        model_C_B_A_1 = Model8(num_categories)
        
        model_B_A_C_1 = Model9(num_categories)
        model_C_A_B_1 = Model10(num_categories)
        
        model_A_C_B_1 = Model11(num_categories)
        model_B_C_A_1 = Model12(num_categories)


        super(StructuralModel, self).__init__(model_A_B_C, model_C_B_A, model_B_A_C, model_C_A_B, model_A_C_B, model_B_C_A, model_A_B_C_1, model_C_B_A_1, model_B_A_C_1, model_C_A_B_1, model_A_C_B_1, model_B_C_A_1)
        self.w = nn.Parameter(torch.tensor(0., dtype=torch.float64))
    
    def set_analytical_maximum_likelihood(self, pi_A, pi_B_A, pi_C_B ):
        self.model_A_B_C.set_analytical_maximum_likelihood(pi_A, pi_B_A, pi_C_B)
        self.model_C_B_A.set_analytical_maximum_likelihood(pi_A, pi_B_A, pi_C_B)
        self.model_B_A_C.set_analytical_maximum_likelihood(pi_A, pi_B_A, pi_C_B)
        self.model_C_A_B.set_analytical_maximum_likelihood(pi_A, pi_B_A, pi_C_B)
        self.model_A_C_B.set_analytical_maximum_likelihood(pi_A, pi_B_A, pi_C_B)
        self.model_B_C_A.set_analytical_maximum_likelihood(pi_A, pi_B_A, pi_C_B)
        
        self.model_A_B_C_1.set_analytical_maximum_likelihood(pi_A, pi_B_A, pi_C_B)
        self.model_C_B_A_1.set_analytical_maximum_likelihood(pi_A, pi_B_A, pi_C_B)
        self.model_B_A_C_1.set_analytical_maximum_likelihood(pi_A, pi_B_A, pi_C_B)
        self.model_C_A_B_1.set_analytical_maximum_likelihood(pi_A, pi_B_A, pi_C_B)
        self.model_A_C_B_1.set_analytical_maximum_likelihood(pi_A, pi_B_A, pi_C_B)
        self.model_B_C_A_1.set_analytical_maximum_likelihood(pi_A, pi_B_A, pi_C_B)

    def set_maximum_likelihood(self, inputs):
        self.model_A_B_C.set_maximum_likelihood(inputs)
        self.model_C_B_A.set_maximum_likelihood(inputs)
        self.model_B_A_C.set_maximum_likelihood(inputs)
        self.model_C_A_B.set_maximum_likelihood(inputs)
        self.model_A_C_B.set_maximum_likelihood(inputs)
        self.model_B_C_A.set_maximum_likelihood(inputs)
        
        self.model_A_B_C_1.set_maximum_likelihood(inputs)
        self.model_C_B_A_1.set_maximum_likelihood(inputs)
        self.model_B_A_C_1.set_maximum_likelihood(inputs)
        self.model_C_A_B_1.set_maximum_likelihood(inputs)
        self.model_A_C_B_1.set_maximum_likelihood(inputs)
        self.model_B_C_A_1.set_maximum_likelihood(inputs)

    def reset_modules_parameters(self):
        self.model_A_B_C.init_parameters()
        self.model_C_B_A.init_parameters()
        self.model_B_A_C.init_parameters()
        self.model_C_A_B.init_parameters()
        self.model_A_C_B.init_parameters()
        self.model_B_C_A.init_parameters()
        
        self.model_A_B_C_1.init_parameters()
        self.model_C_B_A_1.init_parameters()
        self.model_B_A_C_1.init_parameters()
        self.model_C_A_B_1.init_parameters()
        self.model_A_C_B_1.init_parameters()
        self.model_B_C_A_1.init_parameters()
        

