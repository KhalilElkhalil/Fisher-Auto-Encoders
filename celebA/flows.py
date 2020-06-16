'''
This code was taken from: https://github.com/ikostrikov/pytorch-flows/blob/master/flows.py
'''


from torch.autograd import Variable
from torch.autograd import grad
import math
import types

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output
    
    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class MaskedLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 mask,
                 cond_in_features=None,
                 bias=True):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(
                cond_in_features, out_features, bias=False)

        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None):
        output = F.linear(inputs, self.linear.weight * self.mask,
                          self.linear.bias)
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
        return output


nn.MaskedLinear = MaskedLinear


class MADESplit(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509s).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 s_act='tanh',
                 t_act='relu',
                 pre_exp_tanh=False):
        super(MADESplit, self).__init__()

        self.pre_exp_tanh = pre_exp_tanh

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}

        input_mask = get_mask(num_inputs, num_hidden, num_inputs,
                              mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(num_hidden, num_inputs, num_inputs,
                               mask_type='output')

        act_func = activations[s_act]
        self.s_joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask,
                                      num_cond_inputs)

        self.s_trunk = nn.Sequential(act_func(),
                                   nn.MaskedLinear(num_hidden, num_hidden,
                                                   hidden_mask), act_func(),
                                   nn.MaskedLinear(num_hidden, num_inputs,
                                                   output_mask))

        act_func = activations[t_act]
        self.t_joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask,
                                      num_cond_inputs)

        self.t_trunk = nn.Sequential(act_func(),
                                   nn.MaskedLinear(num_hidden, num_hidden,
                                                   hidden_mask), act_func(),
                                   nn.MaskedLinear(num_hidden, num_inputs,
                                                   output_mask))
        
    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.s_joiner(inputs, cond_inputs)
            m = self.s_trunk(h)
            
            h = self.t_joiner(inputs, cond_inputs)
            a = self.t_trunk(h)

            if self.pre_exp_tanh:
                a = torch.tanh(a)
            
            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.s_joiner(x, cond_inputs)
                m = self.s_trunk(h)

                h = self.t_joiner(x, cond_inputs)
                a = self.t_trunk(h)

                if self.pre_exp_tanh:
                    a = torch.tanh(a)

                x[:, i_col] = inputs[:, i_col] * torch.exp(
                    a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)

class MADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509s).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 act='relu',
                 pre_exp_tanh=False):
        super(MADE, self).__init__()

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]

        input_mask = get_mask(
            num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(
            num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask,
                                      num_cond_inputs)

        self.trunk = nn.Sequential(act_func(),
                                   nn.MaskedLinear(num_hidden, num_hidden,
                                                   hidden_mask), act_func(),
                                   nn.MaskedLinear(num_hidden, num_inputs * 2,
                                                   output_mask))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.joiner(inputs, cond_inputs)
            m, a = self.trunk(h).chunk(2, 1)
            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.joiner(x, cond_inputs)
                m, a = self.trunk(h).chunk(2, 1)
                x[:, i_col] = inputs[:, i_col] * torch.exp(
                    a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            s = torch.sigmoid
            return s(inputs), torch.log(s(inputs) * (1 - s(inputs))).sum(
                -1, keepdim=True)
        else:
            return torch.log(inputs /
                             (1 - inputs)), -torch.log(inputs - inputs**2).sum(
                                 -1, keepdim=True)


class Logit(Sigmoid):
    def __init__(self):
        super(Logit, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return super(Logit, self).forward(inputs, 'inverse')
        else:
            return super(Logit, self).forward(inputs, 'direct')


class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (
                    inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
                -1, keepdim=True)


class ActNorm(nn.Module):
    """ An implementation of a activation normalization layer
    from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(ActNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_inputs))
        self.bias = nn.Parameter(torch.zeros(num_inputs))
        self.initialized = False

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if self.initialized == False:
            self.weight.data.copy_(torch.log(1.0 / (inputs.std(0) + 1e-12)))
            self.bias.data.copy_(inputs.mean(0))
            self.initialized = True

        if mode == 'direct':
            return (
                inputs - self.bias) * torch.exp(self.weight), self.weight.sum(
                    -1, keepdim=True).unsqueeze(0).repeat(inputs.size(0), 1)
        else:
            return inputs * torch.exp(
                -self.weight) + self.bias, -self.weight.sum(
                    -1, keepdim=True).unsqueeze(0).repeat(inputs.size(0), 1)


class InvertibleMM(nn.Module):
    """ An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(InvertibleMM, self).__init__()
        self.W = nn.Parameter(torch.Tensor(num_inputs, num_inputs))
        nn.init.orthogonal_(self.W)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs @ self.W, torch.slogdet(
                self.W)[-1].unsqueeze(0).unsqueeze(0).repeat(
                    inputs.size(0), 1)
        else:
            return inputs @ torch.inverse(self.W), -torch.slogdet(
                self.W)[-1].unsqueeze(0).unsqueeze(0).repeat(
                    inputs.size(0), 1)


class LUInvertibleMM(nn.Module):
    """ An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(LUInvertibleMM, self).__init__()
        self.W = torch.Tensor(num_inputs, num_inputs)
        nn.init.orthogonal_(self.W)
        self.L_mask = torch.tril(torch.ones(self.W.size()), -1)
        self.U_mask = self.L_mask.t().clone()

        P, L, U = sp.linalg.lu(self.W.numpy())
        self.P = torch.from_numpy(P)
        self.L = nn.Parameter(torch.from_numpy(L))
        self.U = nn.Parameter(torch.from_numpy(U))

        S = np.diag(U)
        sign_S = np.sign(S)
        log_S = np.log(abs(S))
        self.sign_S = torch.from_numpy(sign_S)
        self.log_S = nn.Parameter(torch.from_numpy(log_S))

        self.I = torch.eye(self.L.size(0))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if str(self.L_mask.device) != str(self.L.device):
            self.L_mask = self.L_mask.to(self.L.device)
            self.U_mask = self.U_mask.to(self.L.device)
            self.I = self.I.to(self.L.device)
            self.P = self.P.to(self.L.device)
            self.sign_S = self.sign_S.to(self.L.device)

        L = self.L * self.L_mask + self.I
        U = self.U * self.U_mask + torch.diag(
            self.sign_S * torch.exp(self.log_S))
        W = self.P @ L @ U

        if mode == 'direct':
            return inputs @ W, self.log_S.sum().unsqueeze(0).unsqueeze(
                0).repeat(inputs.size(0), 1)
        else:
            return inputs @ torch.inverse(
                W), -self.log_S.sum().unsqueeze(0).unsqueeze(0).repeat(
                    inputs.size(0), 1)


class Shuffle(nn.Module):
    """ An implementation of a shuffling layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Shuffle, self).__init__()
        self.perm = np.random.permutation(num_inputs)
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)


class Reverse(nn.Module):
    """ An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)


class CouplingLayer(nn.Module):
    """ An implementation of a coupling layer
    from RealNVP (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 mask,
                 num_cond_inputs=None,
                 s_act='tanh',
                 t_act='relu'):
        super(CouplingLayer, self).__init__()

        self.num_inputs = num_inputs
        self.mask = mask

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'softplus':SmoothReLU}
        s_act_func = activations[s_act]
        t_act_func = activations[t_act]

        if num_cond_inputs is not None:
            total_inputs = num_inputs + num_cond_inputs
        else:
            total_inputs = num_inputs
            
        self.scale_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden), s_act_func(),
            nn.Linear(num_hidden, num_hidden), s_act_func(),
            nn.Linear(num_hidden, num_inputs))
        self.translate_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden), t_act_func(),
            nn.Linear(num_hidden, num_hidden), t_act_func(),
            nn.Linear(num_hidden, num_inputs))

        def init(m):
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0)
                nn.init.orthogonal_(m.weight.data)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        mask = self.mask
        
        masked_inputs = inputs * mask
        if cond_inputs is not None:
            masked_inputs = torch.cat([masked_inputs, cond_inputs], -1)
        
        if mode == 'direct':
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(log_s)
            return inputs * s + t, log_s.sum(-1, keepdim=True)
        else:
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(-log_s)
            return (inputs - t) * s, -log_s.sum(-1, keepdim=True)


class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet


        return inputs, logdets

    def log_probs(self, inputs, cond_inputs = None):
        device = next(self.parameters()).device
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        u, log_jacob = self(inputs, cond_inputs)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)

        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(cond_inputs.shape[0], self.num_inputs).normal_().double()

        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, cond_inputs, mode='inverse')[0]
        return samples

    def sample_pdf(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        u, _ = self(noise, cond_inputs,mode='direct')
        ux = grad(u,noise,create_graph=True,grad_outputs=torch.ones_like(u))[0]
        pdf = (-0.5*u**2)*torch.abs(ux)
        return pdf

class PDFFlowSequential(nn.Sequential):

    def forward(self, t_input, x=None, mode='direct'):

        assert mode in ['direct', 'inverse','pdf']

        if type(x) == type(None):
            t = t_input[:,0].clone()
            in_x = t_input[:,1].clone()
        else:
            in_x = x
            
        if mode == 'direct':
            in_module = list(self._modules.values())[0]
            x = in_module(t,in_x,'direct')
            for module in list(self._modules.values())[1:]:
                x = module(t,x,'direct')

        elif mode == 'pdf':
            in_module = list(self._modules.values())[0]
            x = in_module(t,in_x,'direct')

            for module in list(self._modules.values())[1:]:
                x = module(t,x,'direct')

            f_x = grad(x,in_x,create_graph=True,grad_outputs=torch.ones_like(x))[0].unsqueeze(1)

            if len(x.shape) == 2:
                pdf_vals = torch.exp(-1/2*(x/t)**2)*torch.abs(f_x)
            else:
                pdf_vals = torch.exp(-1/2*(x.unsqueeze(1)/t.unsqueeze(1))**2)*torch.abs(f_x)
            return pdf_vals

        else:
            in_module = list(reversed(self._modules.values()))[0]
            x = in_module(t,in_x,mode)
            for module in list(reversed(self._modules.values()))[1:]:
                x= module(t,x,mode)
        return x 

    def sample_pdf(self,t,num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        samples = self.forward(noise, mode='direct')
        return samples

class PDFFlowLayer(nn.Module):
    def __init__(self):
        super(PDFFlowLayer,self).__init__()

        #self.a  = nn.Parameter(torch.ones(1),requires_grad=True)
        self.a  = nn.Parameter(torch.exp(torch.randn(1)),requires_grad=True)
        self.b  = nn.Parameter(torch.rand(1),requires_grad=True)
        self.c  = nn.Parameter(torch.rand(1),requires_grad=True)
        #self.b  = nn.Parameter(torch.normal(0,torch.ones(1)),requires_grad=True)
        #self.c  = nn.Parameter(torch.normal(0,torch.ones(1)),requires_grad=True)
        #self.a_  = nn.Parameter(torch.ones(1),requires_grad=True)
        self.a_  = nn.Parameter(torch.exp(torch.randn(1)),requires_grad=True)
        #self.b_  = nn.Parameter(torch.normal(0,torch.ones(1)),requires_grad=True)
        #self.c_  = nn.Parameter(torch.normal(0,torch.ones(1)),requires_grad=True)
        self.b_  = nn.Parameter(torch.rand(1),requires_grad=True)
        self.c_  = nn.Parameter(torch.rand(1),requires_grad=True)
        self.beta  = nn.Parameter(torch.normal(0,torch.ones(1)),requires_grad=True)
        self.alpha = nn.Parameter(torch.normal(0,torch.ones(1)),requires_grad=True)

    def activation(self,x,mode='direct'):
        assert mode in ['direct','inverse']
        if mode == 'direct':
            return (self.alpha**2+1)*x+self.alpha*torch.sqrt((self.alpha**2+2)*x**2 + self.beta**2)
        else:
            return (self.alpha**2+1)*x-self.alpha*torch.sqrt((self.alpha**2+2)*x**2 + self.beta**2)
        

    def forward(self,t,x,mode='direct'):
        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            return self.a_*self.activation(self.a*x+self.b*t+self.c,mode=mode)+self.b_*t+self.c_
        else:
            return 1/self.a*self.activation(1/self.a_*x-self.b_/self.a_*t - self.c_/self.a_,mode=mode)-self.b/self.a*t-self.c/self.a

        '''
        if mode == 'direct':
            y = 1/self.my*((self.a**2+1)*(self.mx*x-self.x0)-self.a*torch.sqrt((self.a**2+2)*(self.mx*x-self.x0)**2 +1)) + self.y0
        else:
            y = 1/self.mx*((self.a**2+1)*(self.my*x-self.y0)-self.a*torch.sqrt((self.a**2+2)*(self.my*x-self.y0)**2 +1)) + self.x0
        return y
        '''


class InvertibleLinearMap(nn.Module):
    def __init__(self):
        super(InvertibleLinearMap,self).__init__()
        self.a  = nn.Parameter(torch.exp(torch.randn(1)),requires_grad=True)
        self.b  = nn.Parameter(torch.randn(1),requires_grad=True)
        self.c  = nn.Parameter(torch.randn(1),requires_grad=True)

    def forward(self,t,x,mode='direct'):
        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            return self.a*x+(self.b*t+self.c)
        else:
            return (x-(self.b*t+self.c))/self.a


class PDFFlowLayer2(nn.Module):

    def __init__(self):
        super(PDFFlowLayer2,self).__init__()
        self.beta  = nn.Parameter(torch.rand(1),requires_grad=True)
        self.alpha = nn.Parameter(torch.randn(1),requires_grad=True)


    def forward(self,t,x,mode='direct'):
        assert mode in ['direct','inverse']
        if mode == 'direct':
            return (self.alpha**2+1)*x+self.alpha*torch.sqrt((self.alpha**2+2)*x**2 + self.beta**2)
        else:
            return (self.alpha**2+1)*x-self.alpha*torch.sqrt((self.alpha**2+2)*x**2 + self.beta**2)


class PDFFlow(nn.Module):
    def __init__(self,n_layers=5):
        super(PDFFlow,self).__init__()
        modules = []
        for _ in range(n_layers):
            modules.append(InvertibleLinearMap())
            modules.append(PDFFlowLayer2())
        self.net = PDFFlowSequential(*modules)
    def forward(self,t,x=None):
        return self.net(t,x)

class SmoothReLU(nn.Module):
    def __init__(self):
        super(SmoothReLU,self).__init__()

    def forward(self, x):
        return -F.logsigmoid(-x)
    
class SolveMemory(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, A, x, data=None):
        
        if (data is None):
            data = []
        if (data == []) or torch.any(data[0] != A):
            data.clear()
            data.append(A.data.clone())
            data.append(torch.lu(A.view(1,*A.shape)))
            data.append(torch.lu(A.T.view(1,*A.shape)))
        #else:
        #    print('data saved')
        #print(data)
        ctx.data = data
        solve_x = torch.lu_solve(x.view(1,*x.shape),*data[1])
        solve_x = solve_x.view(solve_x.shape[1:])
        ctx.save_for_backward(A, solve_x)
        return solve_x
    
    @staticmethod
    def backward(ctx, grad_out):
        A, solve_x = ctx.saved_tensors
        data = ctx.data
        dataT = [data[0].T, data[2], data[1]]
        solve_grad = SolveMemory.apply(A.T,grad_out, dataT)
        if ctx.needs_input_grad[0]:
            grad_A = - solve_grad @ solve_x.T
        return grad_A, solve_grad, None

solvememory = SolveMemory.apply

class LinearDegenerate(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(LinearDegenerate, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if in_features>0:
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            self.weight.data /= torch.sqrt(torch.tensor(out_features).double())
        else:
            self.register_parameter('weight', None)
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
            self.bias.data /= torch.sqrt(torch.tensor(out_features).double())
        else:
            self.register_parameter('bias', None)
            
    def forward(self, input):
        if self.weight is not None:
            return F.linear(input, self.weight, self.bias)
        elif self.bias is not None:
            return self.bias
        else:
            return torch.zeros(self.out_features)

    def extra_repr(self):
        return 'in={}, out={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
class InvertibleLinearMap2(nn.Module):
    def __init__(self, dim):
        super(InvertibleLinearMap2,self).__init__()
        self.A  = nn.Parameter(torch.randn(dim,dim).double(),requires_grad=True)
        self.data = []

    def forward(self,x,mode='direct'):
        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            return x.mm(self.A)
        else:
            return solvememory(self.A.T,x.T,self.data).T 
            

class InvertibleLinearMap3(nn.Module):
    def __init__(self, dim):
        super(InvertibleLinearMap,self).__init__()
        self.A  = nn.Parameter(torch.randn(dim,dim),requires_grad=True)
        

    def forward(self,x,mode='direct'):
        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            return x.mm(self.A)
        else:
            return torch.solve(x.T, self.A.T)[0].T

class AddLinearConditional(nn.Module):

    def __init__(self, ninputs, ncond_inputs, bias=True):
        super(AddLinearConditional, self).__init__()
        self.ninputs = ninputs
        self.ncond_inputs = ncond_inputs
        if ncond_inputs>0:
            self.weight = nn.Parameter(torch.Tensor(ncond_inputs,ninputs))
        else:
            self.register_parameter('weight', None)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ninputs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            if param is not None:
                if param == self.bias:
                    nn.init.normal_(param)
                else:
                    nn.init.xavier_normal_(param)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if self.weight is not None:
            out = F.linear(cond_inputs, self.weight, self.bias)
        elif self.bias is not None:
            out = self.bias
        else:
            return inputs, 0
        if mode == 'direct':
            return inputs + out, 0 
        else:
            return inputs - out, 0 

class LUInvertibleMM(nn.Module):

    def __init__(self, num_inputs):
        super(LUInvertibleMM, self).__init__()
        self.dim = num_inputs
        self.I = torch.eye(num_inputs).to('cpu')#"cuda:0" if torch.cuda.is_available() else "cpu")
        self.LU =  nn.Parameter(torch.Tensor(num_inputs, num_inputs))
        self.reset_parameters()

    def reset_parameters(self):
        W = torch.torch.empty_like(self.LU)
        nn.init.eye_(W)
        LU, P = W.lu()
        self.LU.data =  LU
        P = torch.lu_unpack(LU, P, unpack_data=False)[0]
        self.P = torch.nonzero(P)[:,1].to('cpu')#"cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, inputs, cond_inputs=None, mode='direct'):

        logdet = torch.sum(self.LU.diag().abs().log())

        if mode == 'direct':
            if self.dim > 1:
                W = self.LU.tril(-1)
                W += self.I
                W @= self.LU.triu()
                return inputs @ torch.index_select(W, 0, self.P), logdet
            else:
                return inputs * self.LU, logdet

        else:
            if self.dim > 1:
                output = torch.triangular_solve(inputs.T,self.LU,
                                                transpose=True).solution
                output = torch.triangular_solve(output,self.LU,
                                                upper=False,
                                                unitriangular=True,
                                                transpose=True).solution
                return torch.index_select(output.T, 1, self.P), -logdet
            else:
                return inputs / self.LU, -logdet

class PiecewiseReLU(nn.Module):
    def __init__(self):
        super(PiecewiseReLU,self).__init__()
    def forward(self, x, cond_inputs=None, mode='direct'):
        assert mode in ['direct','inverse']
        if mode == 'direct':
            out = torch.where(x>0,x,(1-x).log())
        else:
            out = torch.where(x>0,x,1-torch.exp(-x))
        dx_pt = torch.autograd.grad(out,x,create_graph=True,grad_outputs=torch.ones_like(out))[0]
        return out, dx_pt.log().sum(dim=-1,keepdims=True)

class HyperReLU(nn.Module):
    def __init__(self):
        super(HyperReLU,self).__init__()
        self.beta  = nn.Parameter(torch.rand(1))
        self.alpha = nn.Parameter(torch.randn(1)/3)

    def forward(self, x, cond_inputs=None, mode='direct'):
        assert mode in ['direct','inverse']
        if mode == 'direct':
            out = (self.alpha**2+1)*x+self.alpha*torch.sqrt((self.alpha**2+2)*x**2 + self.beta**2)
        else:
            out = (self.alpha**2+1)*x-self.alpha*torch.sqrt((self.alpha**2+2)*x**2 + self.beta**2)
        dx_pt = torch.autograd.grad(out,x,create_graph=True,grad_outputs=torch.ones_like(out))[0]
        return out, torch.sum(torch.log(dx_pt),dim=-1,keepdim=True)

class PDFFlowLayer3(nn.Module):

    def __init__(self,ndims,nparams):
        super(PDFFlowLayer2,self).__init__()
        self.out_fcn = HyperReLU()
        self.x_fcn = InvertibleLinearMap2(ndims)
        self.t_fcn = LinearDegenerate(nparams,ndims)
                              
    def forward(self,t,x,mode='direct'):
        assert mode in ['direct','inverse']
        if mode == 'direct':
            return self.out_fcn(self.x_fcn(x)+self.t_fcn(t))
        else:
            return self.x_fcn(self.out_fcn(x,'inverse')-self.t_fcn(t),'inverse')
