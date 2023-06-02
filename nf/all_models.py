import math
import numpy as np
import scipy as sp
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from nf.utils import unconstrained_RQS
from torch.distributions import HalfNormal, Weibull, Gumbel


class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """

    def __init__(self, in_dim, out_dim, hidden_dim, activation="tanh"):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, x):
        return self.network(x)


class SumGaussModel(nn.Module):
    """
    This function is for the quantization of the halo field. That is it models the probability of 
    observing number of halos in a given voxel as a sum of gausians.
    """

    def __init__(
            self,
            dim=1,
            hidden_dim=8,
            base_network=FCNN,
            num_cond=0,
            ngauss=1,
            mu_all=None,
            sig_all=None,
            base_dist='pl_exp'
        ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        self.num_cond = num_cond
        self.ngauss = ngauss
        self.base_dist = base_dist
        if mu_all is not None:
            self.mu_all = torch.tensor(mu_all, device='cuda')
        else:
            self.mu_all = mu_all
        if sig_all is not None:
            self.sig_all = torch.tensor(sig_all, device='cuda')
            self.var_all = torch.tensor(sig_all**2, device='cuda')
        else:
            self.sig_all = sig_all
            self.var_all = None

        if (self.mu_all is None) or (self.sig_all is None):
            self.layer_init = base_network(self.num_cond, 3 * self.ngauss, hidden_dim)
        else:
            if base_dist is None:
                self.layer_init = base_network(self.num_cond, self.ngauss, hidden_dim)
            elif base_dist == 'pl_exp':
                self.layer_init = base_network(self.num_cond, self.ngauss + 2, hidden_dim)
            else:
                raise ValueError("base_dist not supported")

        if self.num_cond == 0:
            self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.initial_param, -math.sqrt(0.5), math.sqrt(0.5))

    def forward(self, x, cond_inp=None):
        out = self.layer_init(cond_inp)
        if (self.mu_all is None) or (self.sig_all is None):
            mu_all, alpha_all, pw_all = (
                out[:, 0:self.ngauss],
                out[:, self.ngauss:2 * self.ngauss],
                out[:, 2 * self.ngauss:3 * self.ngauss],
                )
            mu_all = (1 + nn.Tanh()(mu_all)) / 2
            var_all = torch.exp(alpha_all)
            pw_all = nn.Softmax(dim=1)(pw_all)
            Li_all = torch.zeros(mu_all.shape[0])
            Li_all = Li_all.to('cuda')
            for i in range(self.ngauss):
                Li_all += (
                    pw_all[:, i] * (1 / torch.sqrt(2 * np.pi * var_all[:, i])) *
                    torch.exp(-0.5 * ((x[:, 0] - mu_all[:, i])**2) / (var_all[:, i]))
                    )
            logP = torch.log(Li_all)
        else:
            mu_all, var_all = self.mu_all, self.var_all
            if self.base_dist is None:
                pw_all_inp = self.layer_init(cond_inp)
            elif self.base_dist == 'pl_exp':
                # in this we have a power law and an exponential as a base distribution
                out = self.layer_init(cond_inp)
                out = torch.exp(out)
                pw_all_orig = out[:, 0:self.ngauss]
                al = out[:, self.ngauss]
                # put al between 0 and 2
                al = 0. * nn.Tanh()(al)
                bt = out[:, self.ngauss + 1] + 1.
                # we first predict the base distirbution given the alpha and beta of the form mu**alpha * exp(-beta*mu)
                base_pws = torch.zeros(x.shape[0], self.ngauss)
                base_pws = base_pws.to('cuda')
                for i in range(self.ngauss):
                    base_pws[:, i] = torch.pow(mu_all[i], al) * torch.exp(-bt * mu_all[i])
                pw_all_inp = torch.mul(pw_all_orig, base_pws)
            else:
                raise ValueError("base_dist not supported")

            pw_all = nn.Softmax(dim=1)(pw_all_inp)
            Li_all = torch.zeros(x.shape[0])
            Li_all = Li_all.to('cuda')
            for i in range(self.ngauss):
                Li_all += (
                    pw_all[:, i] * (1 / torch.sqrt(2 * np.pi * var_all[i])) *
                    torch.exp(-0.5 * ((x[:, 0] - mu_all[i])**2) / (var_all[i]))
                    )

            logP = torch.log(Li_all + 1e-30)
        return logP

    def inverse(self, cond_inp=None):
        #
        if (self.mu_all is None) or (self.sig_all is None):
            out = self.layer_init(cond_inp)
            mu_all, alpha_all, pw_all = (
                out[:, 0:self.ngauss],
                out[:, self.ngauss:2 * self.ngauss],
                out[:, 2 * self.ngauss:3 * self.ngauss],
                )
            mu_all = (1 + nn.Tanh()(mu_all)) / 2
            pw_all = nn.Softmax(dim=1)(pw_all)
            var_all = torch.exp(alpha_all)
            counts = torch.distributions.multinomial.Multinomial(total_count=1, probs=pw_all).sample()
            counts = counts.to('cuda')
            # loop over gaussians
            z = torch.empty(0, device=counts.device)
            for k in range(self.ngauss):
                # find indices where count is non-zero for kth gaussian
                ind = torch.nonzero(counts[:, k])
                # if there are any indices, sample from kth gaussian
                if ind.shape[0] > 0:
                    z_k = (mu_all[ind, k][:, 0] + torch.randn(ind.shape[0]) * torch.sqrt(var_all[ind, k])[:, 0])
                    z = torch.cat((z, z_k), dim=0)

        else:
            if self.base_dist is None:
                pw_all = self.layer_init(cond_inp)
            elif self.base_dist == 'pl_exp':
                # in this we have a power law and an exponential as a base distribution
                out = self.layer_init(cond_inp)
                out = torch.exp(out)
                pw_all_orig = out[:, 0:self.ngauss]
                al = out[:, self.ngauss]
                al = 0. * nn.Tanh()(al)
                bt = out[:, self.ngauss + 1] + 1.
                # we first predict the base distirbution given the alpha and beta of the form mu**alpha * exp(-beta*mu)
                base_pws = torch.zeros(out.shape[0], self.ngauss)
                base_pws = base_pws.to('cuda')
                for i in range(self.ngauss):
                    base_pws[:, i] = torch.pow(self.mu_all[i], al) * torch.exp(-bt * self.mu_all[i])
                pw_all = torch.mul(pw_all_orig, base_pws)
            pw_all = nn.Softmax(dim=1)(pw_all)

            var_all = self.var_all
            mu_all = self.mu_all

            counts = torch.distributions.multinomial.Multinomial(total_count=1, probs=pw_all).sample()
            counts = counts.to('cuda')
            # loop over gaussians
            # z = torch.empty(0, device=counts.device)
            z_out = torch.empty(counts.shape[0], device=counts.device)
            for k in range(self.ngauss):
                # find indices where count is non-zero for kth gaussian
                ind = torch.nonzero(counts[:, k])
                # if there are any indices, sample from kth gaussian
                if ind.shape[0] > 0:
                    z_k = (mu_all[k] + torch.randn(ind.shape[0], device='cuda') * torch.sqrt(var_all[k]))
                    z_out[ind[:, 0]] = z_k
                    # z = torch.cat((z, z_k), dim=0)
        return z_out

    def sample(self, cond_inp=None, mask=None):
        x = self.inverse(cond_inp, mask)
        return x


class NSF_M1_CNNcond(nn.Module):
    """
    This function models the probability of observing the heaviest halo mass given the density field.
    """

    def __init__(
        self,
        dim=1,
        K=5,
        B=3,
        hidden_dim=8,
        base_network=FCNN,
        num_cond=0,
        nflows=1,
        ngauss=1,
        base_dist="gauss",
        mu_pos=False,
        ):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.num_cond = num_cond
        self.nflows = nflows
        self.ngauss = ngauss
        self.base_dist = base_dist
        self.mu_pos = mu_pos
        self.num_cond = num_cond
        self.init_param = nn.Parameter(torch.Tensor(3 * K - 1))
        if self.base_dist in ["gauss", "halfgauss"]:
            if self.ngauss == 1:
                self.layer_init_gauss = base_network(self.num_cond, 2, hidden_dim)
            else:
                self.layer_init_gauss = base_network(self.num_cond, 3 * self.ngauss, hidden_dim)
        elif self.base_dist == 'weibull':
            self.layer_init_gauss = base_network(self.num_cond, 2, hidden_dim)
        elif self.base_dist == 'gumbel':
            self.layer_init_gauss = base_network(self.num_cond, 2, hidden_dim)
        else:
            print('base_dist not recognized')
            raise ValueError

        self.layers = nn.ModuleList()
        for jf in range(nflows):
            self.layers += [base_network(self.num_cond, 3 * K - 1, hidden_dim)]

        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.init_param, -1 / 2, 1 / 2)

    def get_gauss_func_mu_alpha(self, cond_inp=None):
        out = self.layer_init_gauss(cond_inp)
        if self.ngauss == 1:
            mu, alpha = out[:, 0], out[:, 1]
            if self.mu_pos:
                mu = (1 + nn.Tanh()(mu)) / 2
            var = torch.exp(alpha)
            return mu, var
        else:
            mu_all, alpha_all, pw_all = (
                out[:, 0:self.ngauss], out[:, self.ngauss:2 * self.ngauss], out[:, 2 * self.ngauss:3 * self.ngauss]
                )
            if self.mu_pos:
                mu_all = (1 + nn.Tanh()(mu_all)) / 2
            pw_all = nn.Softmax(dim=1)(pw_all)
            var_all = torch.exp(alpha_all)
            return mu_all, var_all, pw_all

    def forward(self, x, cond_inp=None):
        if self.base_dist in ["gauss", "halfgauss"]:
            if self.ngauss == 1:
                mu, var = self.get_gauss_func_mu_alpha(cond_inp)
            else:
                mu_all, var_all, pw_all = self.get_gauss_func_mu_alpha(cond_inp)
        elif self.base_dist in ['weibull', 'gumbel']:
            out = self.layer_init_gauss(cond_inp)
            mu, alpha = out[:, 0], out[:, 1]
            if self.base_dist == 'weibull':
                scale, conc = torch.exp(mu), torch.exp(alpha)
            else:
                if self.mu_pos:
                    mu = torch.exp(mu)
                    # mu = (1 + nn.Tanh()(mu)) / 2
                sig = torch.exp(alpha)
        else:
            print('base_dist not recognized')
            raise ValueError

        if len(x.shape) > 1:
            x = x[:, 0]
        log_det_all = torch.zeros_like(x)
        for jf in range(self.nflows):
            out = self.layers[jf](cond_inp)
            z = torch.zeros_like(x)
            # log_det_all = torch.zeros(z.shape)
            W, H, D = torch.split(out, self.K, dim=1)
            W, H = torch.softmax(W, dim=1), torch.softmax(H, dim=1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            z, ld = unconstrained_RQS(x, W, H, D, inverse=False, tail_bound=self.B)
            log_det_all += ld
            x = z

        if self.base_dist == 'gauss':
            if self.ngauss == 1:
                logp = -0.5 * np.log(2 * np.pi) - 0.5 * torch.log(var) - 0.5 * (x - mu)**2 / var
            else:
                Li_all = torch.zeros(mu_all.shape[0])
                Li_all = Li_all.to('cuda')
                for i in range(self.ngauss):
                    Li_all += (
                        pw_all[:, i] * (1 / torch.sqrt(2 * np.pi * var_all[:, i])) *
                        torch.exp(-0.5 * ((x - mu_all[:, i])**2) / (var_all[:, i]))
                        )
                logp = torch.log(Li_all)

        elif self.base_dist == 'halfgauss':
            if self.ngauss == 1:
                x = torch.exp(x - mu)
                hf = HalfNormal((torch.sqrt(var)))
                logp = hf.log_prob(x)

        elif self.base_dist == 'weibull':
            hf = Weibull(scale, conc)
            logp = hf.log_prob(x)
            # if there are any nans of infs, replace with -100
            logp[torch.isnan(logp) | torch.isinf(logp)] = -100
        elif self.base_dist == 'gumbel':
            hf = Gumbel(mu, sig)
            logp = hf.log_prob(x)
            logp[torch.isnan(logp) | torch.isinf(logp)] = -100
        else:
            raise ValueError("Base distribution not supported")

        logp = log_det_all + logp
        return logp

    def inverse(self, cond_inp=None, mask=None):
        if self.base_dist in ["gauss", "halfgauss"]:
            if self.ngauss == 1:
                mu, var = self.get_gauss_func_mu_alpha(cond_inp)
            else:
                mu_all, var_all, pw_all = self.get_gauss_func_mu_alpha(cond_inp)
        elif self.base_dist in ['weibull', 'gumbel']:
            out = self.layer_init_gauss(cond_inp)
            mu, alpha = out[:, 0], out[:, 1]
            if self.base_dist == 'weibull':
                scale, conc = torch.exp(mu), torch.exp(alpha)
            else:
                if self.mu_pos:
                    mu = torch.exp(mu)
                    # mu = (1 + nn.Tanh()(mu)) / 2
                sig = torch.exp(alpha)
        else:
            print('base_dist not recognized')
            raise ValueError

        if self.base_dist == 'gauss':
            if self.ngauss == 1:
                x = mu + torch.randn(cond_inp.shape[0]) * torch.sqrt(var)
            else:
                counts = torch.distributions.multinomial.Multinomial(total_count=1, probs=pw_all).sample()
                # loop over gaussians
                x = torch.empty(0, device=counts.device)
                for k in range(self.ngauss):
                    # find indices where count is non-zero for kth gaussian
                    ind = torch.nonzero(counts[:, k])
                    # if there are any indices, sample from kth gaussian
                    if ind.shape[0] > 0:
                        x_k = (mu_all[ind, k][:, 0] + torch.randn(ind.shape[0]) * torch.sqrt(var_all[ind, k])[:, 0])
                        x = torch.cat((x, x_k), dim=0)

        if self.base_dist == 'halfgauss':
            if self.ngauss == 1:
                x = torch.log(mu + torch.abs(torch.randn(cond_inp.shape[0])) * torch.sqrt(var))

        if self.base_dist == 'weibull':
            hf = Weibull(scale, conc)
            x = hf.sample()
        if self.base_dist == 'gumbel':
            hf = Gumbel(mu, sig)
            x = hf.sample()

        log_det_all = torch.zeros_like(x)
        for jf in range(self.nflows):
            ji = self.nflows - jf - 1
            out = self.layers[ji](cond_inp)
            z = torch.zeros_like(x)
            W, H, D = torch.split(out, self.K, dim=1)
            W, H = torch.softmax(W, dim=1), torch.softmax(H, dim=1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            z, ld = unconstrained_RQS(x, W, H, D, inverse=True, tail_bound=self.B)
            log_det_all += ld
            x = z

        x *= mask[:, 0]
        return x, log_det_all

    def sample(self, cond_inp=None, mask=None):
        x, _ = self.inverse(cond_inp, mask)
        return x


class NSF_Mdiff_CNNcond(nn.Module):
    """
    This function models the probability of observing all the lower halo masses
    """

    def __init__(
        self,
        dim=None,
        K=5,
        B=3,
        hidden_dim=8,
        base_network=FCNN,
        num_cond=0,
        nflows=1,
        ngauss=1,
        base_dist="gumbel",
        mu_pos=False,
        ):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.num_cond = num_cond
        self.nflows = nflows
        self.ngauss = ngauss
        self.base_dist = base_dist
        self.mu_pos = mu_pos
        self.num_cond = num_cond
        self.init_param = nn.Parameter(torch.Tensor(3 * K - 1))
        self.layers_all_dim = nn.ModuleList()
        self.layers_all_dim_init = nn.ModuleList()
        # self.layers_all_dim = []
        # self.layers_all_dim_init = []
        for jd in range(dim):
            if self.base_dist in ["gauss", "halfgauss"]:
                if self.ngauss == 1:
                    layer_init_gauss = base_network(self.num_cond + jd, 2, hidden_dim)
                else:
                    layer_init_gauss = base_network(self.num_cond + jd, 3 * self.ngauss, hidden_dim)
            elif self.base_dist == 'weibull':
                layer_init_gauss = base_network(self.num_cond + jd, 2, hidden_dim)
            elif self.base_dist == 'gumbel':
                layer_init_gauss = base_network(self.num_cond + jd, 2, hidden_dim)
            else:
                print('base_dist not recognized')
                raise ValueError
            self.layers_all_dim_init += [layer_init_gauss]

            layers = nn.ModuleList()
            for jf in range(nflows):
                layers += [base_network(self.num_cond + jd, 3 * K - 1, hidden_dim)]
            self.layers_all_dim += [layers]

        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.init_param, -1 / 2, 1 / 2)

    def get_gauss_func_mu_alpha(self, jd, cond_inp=None):
        out = self.layers_all_dim_init[jd](cond_inp)
        if self.ngauss == 1:
            mu, alpha = out[:, 0], out[:, 1]
            if self.mu_pos:
                mu = (1 + nn.Tanh()(mu)) / 2
            var = torch.exp(alpha)
            return mu, var
        else:
            mu_all, alpha_all, pw_all = (
                out[:, 0:self.ngauss], out[:, self.ngauss:2 * self.ngauss], out[:, 2 * self.ngauss:3 * self.ngauss]
                )
            if self.mu_pos:
                mu_all = (1 + nn.Tanh()(mu_all)) / 2
            pw_all = nn.Softmax(dim=1)(pw_all)
            var_all = torch.exp(alpha_all)
            return mu_all, var_all, pw_all

    def forward(self, x_inp, cond_inp=None, mask=None):
        logp = torch.zeros_like(x_inp)
        logp = logp.to('cuda')
        x_inp = x_inp.to('cuda')
        for jd in range(self.dim):
            # print(cond_inp.shape)
            if jd > 0:
                cond_inp_jd = torch.cat([cond_inp, x_inp[:, :jd]], dim=1)
            else:
                cond_inp_jd = cond_inp
            # print(cond_inp.shape)
            if self.base_dist in ["halfgauss"]:
                if self.ngauss == 1:
                    mu, var = self.get_gauss_func_mu_alpha(jd, cond_inp_jd)
            elif self.base_dist in ['weibull', 'gumbel']:
                out = self.layers_all_dim_init[jd](cond_inp_jd)
                mu, alpha = out[:, 0], out[:, 1]
                if self.base_dist == 'weibull':
                    scale, conc = torch.exp(mu), torch.exp(alpha)
                else:
                    if self.mu_pos:
                        # mu = torch.exp(mu)
                        mu = (1 + nn.Tanh()(mu)) / 2
                    sig = torch.exp(alpha)
            else:
                print('base_dist not recognized')
                raise ValueError

            # if len(x.shape) > 1:
            #     x = x[:, 0]
            log_det_all_jd = torch.zeros(x_inp.shape[0])
            log_det_all_jd = log_det_all_jd.to('cuda')
            for jf in range(self.nflows):
                if jf == 0:
                    x = x_inp[:, jd]
                    x = x.to('cuda')
                out = self.layers_all_dim[jd][jf](cond_inp_jd)
                # z = torch.zeros_like(x)
                # log_det_all = torch.zeros(z.shape)
                W, H, D = torch.split(out, self.K, dim=1)
                W, H = torch.softmax(W, dim=1), torch.softmax(H, dim=1)
                W, H = 2 * self.B * W, 2 * self.B * H
                D = F.softplus(D)
                z, ld = unconstrained_RQS(x, W, H, D, inverse=False, tail_bound=self.B)
                log_det_all_jd += ld
                x = z

            if self.base_dist == 'halfgauss':
                if self.ngauss == 1:
                    x = torch.exp(x - mu)
                    hf = HalfNormal((torch.sqrt(var)))
                    logp_jd = hf.log_prob(x)

            elif self.base_dist == 'weibull':
                hf = Weibull(scale, conc)
                logp_jd = hf.log_prob(x)
                # if there are any nans of infs, replace with -100
                logp_jd[torch.isnan(logp_jd) | torch.isinf(logp_jd)] = -100
            elif self.base_dist == 'gumbel':
                hf = Gumbel(mu, sig)
                logp_jd = hf.log_prob(x)
                logp_jd[torch.isnan(logp_jd) | torch.isinf(logp_jd)] = -100
            else:
                raise ValueError("Base distribution not supported")

            logp[:, jd] = log_det_all_jd + logp_jd
        logp *= mask
        logp = torch.sum(logp, dim=1)
        # print(logp.shape, mask.shape)
        return logp

    def inverse(self, cond_inp=None, mask=None):
        z_out = torch.zeros((cond_inp.shape[0], self.dim))
        z_out = z_out.to('cuda')
        for jd in range(self.dim):
            if jd > 0:
                cond_inp_jd = torch.cat([cond_inp, z_out[:, :jd]], dim=1)
            else:
                cond_inp_jd = cond_inp
            if self.base_dist in ["halfgauss"]:
                if self.ngauss == 1:
                    mu, var = self.get_gauss_func_mu_alpha(jd, cond_inp_jd)

            elif self.base_dist in ['gumbel', 'weibull']:
                out = self.layers_all_dim_init[jd](cond_inp_jd)
                mu, alpha = out[:, 0], out[:, 1]
                if self.base_dist == 'weibull':
                    scale, conc = torch.exp(mu), torch.exp(alpha)
                else:
                    if self.mu_pos:
                        # mu = torch.exp(mu)
                        mu = (1 + nn.Tanh()(mu)) / 2
                    sig = torch.exp(alpha)
            else:
                print('base_dist not recognized')
                raise ValueError

            if self.base_dist == 'gauss':
                if self.ngauss == 1:
                    x = mu + torch.randn(cond_inp_jd.shape[0], device='cuda') * torch.sqrt(var)

            elif self.base_dist == 'halfgauss':
                if self.ngauss == 1:
                    x = torch.log(mu + torch.abs(torch.randn(cond_inp_jd.shape[0], device='cuda')) * torch.sqrt(var))

            elif self.base_dist == 'weibull':
                hf = Weibull(scale, conc)
                x = hf.sample()
            elif self.base_dist == 'gumbel':
                hf = Gumbel(mu, sig)
                x = hf.sample()
                # print(x.shape)
                # print(mu, sig)
            else:
                raise ValueError("Base distribution not supported")

            log_det_all = torch.zeros_like(x)
            for jf in range(self.nflows):
                ji = self.nflows - jf - 1
                out = self.layers_all_dim[jd][ji](cond_inp_jd)
                z = torch.zeros_like(x)
                W, H, D = torch.split(out, self.K, dim=1)
                W, H = torch.softmax(W, dim=1), torch.softmax(H, dim=1)
                W, H = 2 * self.B * W, 2 * self.B * H
                D = F.softplus(D)
                z, ld = unconstrained_RQS(x, W, H, D, inverse=True, tail_bound=self.B)
                log_det_all += ld
                x = z

            x *= mask[:, jd]
            z_out[:, jd] = x
        return z_out, log_det_all

    def sample(self, cond_inp=None, mask=None):
        x, _ = self.inverse(cond_inp, mask)
        return x


# class MAF_CNN_cond(nn.Module):
#     """
#     This is the model for the auto-regressive model of the lower halo masses.
#     This takes as input the environment, heavist halo mass and number of halos. 
#     It is based on simple CNNs with auto-regressive structure.
#     """

#     def __init__(self, dim, hidden_dim=8, base_network=FCNN, num_cond=0):
#         super().__init__()
#         self.dim = dim
#         self.layers = nn.ModuleList()
#         self.num_cond = num_cond
#         if self.num_cond == 0:
#             self.initial_param = nn.Parameter(torch.Tensor(1))
#         else:
#             self.layer_init = base_network(self.num_cond, 2, hidden_dim)
#         for i in range(1, dim):
#             self.layers += [base_network(self.num_cond + i, 2, hidden_dim)]
#         self.mu_all_forward = np.zeros(self.dim)
#         self.alpha_all_forward = np.zeros(self.dim)
#         self.mu_all_inverse = torch.zeros(self.dim)
#         self.alpha_all_inverse = torch.zeros(self.dim)
#         if self.num_cond == 0:
#             self.reset_parameters()

#     def reset_parameters(self):
#         init.uniform_(self.initial_param, -math.sqrt(0.5), math.sqrt(0.5))

#     def forward(self, x, cond_inp=None, mask=None):
#         z = torch.zeros_like(x)
#         # log_det = torch.zeros(z.shape[0])
#         log_det_all = torch.zeros_like(x)

#         for i in range(self.dim):
#             if i == 0:
#                 out = self.layer_init(cond_inp)
#                 mu, alpha = out[:, 0], out[:, 1]
#                 # mu = -torch.exp(mu)
#                 mu = (1 + nn.Tanh()(mu))
#             else:
#                 out = self.layers[i - 1](torch.cat([cond_inp, x[:, :i]], dim=1))
#                 mu, alpha = out[:, 0], out[:, 1]
#                 # mu = -torch.exp(mu)
#                 mu = (1 + nn.Tanh()(mu))

#             z[:, i] = (x[:, i] - mu) / torch.exp(alpha)
#             log_det_all[:, i] = -alpha

#             # try:
#             #     self.mu_all_forward[i] = mu.detach().numpy()
#             #     self.alpha_all_forward[i] = alpha.detach().numpy()
#             # except:
#             #     self.mu_all_forward[i] = mu[0].detach().numpy()
#             #     self.alpha_all_forward[i] = alpha[0].detach().numpy()
#         log_det_all_masked = log_det_all * mask
#         log_det = torch.sum(log_det_all_masked, dim=1)
#         return z, log_det

#     def inverse(self, z, cond_inp=None, mask=None):
#         x = torch.zeros_like(z)
#         x = x.to('cuda')
#         z = z.to('cuda')
#         log_det_all = torch.zeros_like(z)
#         log_det_all = log_det_all.to('cuda')
#         for i in range(self.dim):
#             if i == 0:
#                 out = self.layer_init(cond_inp)
#                 mu, alpha = out[:, 0], out[:, 1]
#                 # mu = -torch.exp(mu)
#                 mu = (1 + nn.Tanh()(mu))
#             else:
#                 out = self.layers[i - 1](torch.cat([cond_inp, x[:, :i]], dim=1))
#                 mu, alpha = out[:, 0], out[:, 1]
#                 # mu = -torch.exp(mu)
#                 mu = (1 + nn.Tanh()(mu))

#             x[:, i] = mu + torch.exp(alpha) * z[:, i]
#             log_det_all[:, i] = alpha

#         log_det_all_masked = log_det_all * mask
#         log_det = torch.sum(log_det_all_masked, dim=1)
#         x *= mask
#         return x, log_det
