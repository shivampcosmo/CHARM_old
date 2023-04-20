import torch
import torch.nn as nn
import numpy as np
from nf.cnn_3d_stack import CNN3D_stackout


class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """

    def __init__(self, in_dim, out_dim, hidden_dim):
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


class COMBINED_Model(nn.Module):
    """
    Combined model for the AR_NPE.
    """

    def __init__(
        self,
        priors_all,
        Mdiff_model,
        M1_model,
        Ntot_model,
        ndim,
        ksize,
        nside_in,
        nside_out,
        nbatch,
        ninp,
        nfeature,
        nout,
        layers_types=['cnn', 'res', 'res', 'res'],
        act='tanh',
        padding='valid',
        sep_Ntot_cond=False,
        sep_M1_cond=False,
        sep_Mdiff_cond=False,
        ):
        super().__init__()
        self.priors_all = priors_all
        self.M1_model = M1_model
        self.Ntot_model = Ntot_model
        self.Mdiff_model = Mdiff_model
        self.nbatch = nbatch
        self.nout = nout
        self.ninp = ninp

        self.conv_layers = CNN3D_stackout(
            ksize,
            nside_in,
            nside_out,
            nbatch,
            ninp,
            nfeature,
            nout,
            layers_types=layers_types,
            act=act,
            padding=padding
            )
        self.ndim = ndim
        self.sep_Ntot_cond = sep_Ntot_cond
        self.sep_M1_cond = sep_M1_cond
        self.sep_Mdiff_cond = sep_Mdiff_cond
        if self.sep_Ntot_cond:
            self.cond_Ntot_layer = FCNN(nout + ninp, nout + ninp, nout + ninp)
        if self.sep_M1_cond:
            self.cond_M1_layer = FCNN(nout + ninp + 1, nout + ninp + 1, nout + ninp + 1)
        if self.sep_Mdiff_cond:
            self.cond_Mdiff_layer = FCNN(nout + ninp + 2, nout + ninp + 2, nout + ninp + 2)

    def forward(
        self,
        x_Mdiff,
        x_M1,
        x_Ntot,
        cond_x=None,
        cond_x_nsh=None,
        mask_Mdiff_truth=None,
        mask_M1_truth=None,
        Nhalos_truth=None,
        use_Ntot_samples=False,
        use_M1_samples=False,
        train_Ntot=False,
        train_M1=False,
        train_Mdiff=False,
        ):
        cond_out = self.conv_layers(cond_x)
        cond_out = torch.cat((cond_out, cond_x_nsh), dim=1)

        if self.sep_Ntot_cond:
            cond_out_Ntot = self.cond_Ntot_layer(cond_out)
        else:
            cond_out_Ntot = cond_out
        logP_Ntot = torch.zeros(1, device='cuda')
        if train_Ntot:
            logP_Ntot = self.Ntot_model.forward(x_Ntot, cond_out_Ntot)

            if use_Ntot_samples:
                Ntot_samp = np.maximum(np.round(self.Ntot_model.inverse(cond_out_Ntot).detach().numpy()) - 1,
                                       0).astype(int)
                mask_samp_all = np.zeros((Ntot_samp.shape[0], Ntot_samp.shape[1], self.ndim))
                idx = np.arange(self.ndim)[None, None, :]
                mask_samp_all[np.arange(Ntot_samp.shape[0])[:, None, None],
                              np.arange(Ntot_samp.shape[1])[None, :, None], idx] = (idx < Ntot_samp[..., None])

                Ntot_samp_diff = Ntot_samp - 1
                Ntot_samp_diff[Ntot_samp_diff < 0] = 0
                mask_samp_M_diff = np.zeros((Ntot_samp.shape[0], Ntot_samp.shape[1], self.ndim - 1))
                idx = np.arange(self.ndim - 1)[None, None, :]
                mask_samp_M_diff[np.arange(Ntot_samp.shape[0])[:, None, None],
                                 np.arange(Ntot_samp.shape[1])[None, :, None], idx] = (idx < Ntot_samp_diff[..., None])

                mask_samp_M1 = mask_samp_all[:, :, 0]

                mask_M1_truth = torch.from_numpy(mask_samp_M1).float().cuda()
                mask_Mdiff_truth = torch.from_numpy(mask_samp_M_diff).float().cuda()
                Nhalos_truth = torch.from_numpy(Ntot_samp).float().cuda()
            else:
                # Nhalos_truth = np.maximum(np.round(x_Ntot.cpu().detach().numpy()), 0).astype(int)
                # tensor_zero = torch.Tensor(0).cuda()
                # Nhalos_truth = torch.maximum(torch.round(x_Ntot), tensor_zero)
                Nhalos_truth = Nhalos_truth.to('cuda')
        else:
            Nhalos_truth = Nhalos_truth.to('cuda')

        logP_M1 = torch.zeros(1, device='cuda')
        if train_M1:
            cond_inp_M1 = torch.cat([Nhalos_truth, cond_out], dim=1)
            if self.sep_M1_cond:
                cond_inp_M1 = self.cond_M1_layer(cond_inp_M1)

            logP_M1 = self.M1_model.forward(x_M1, cond_inp_M1)
            logP_M1 *= mask_M1_truth
            if use_M1_samples:
                M1_samp = self.M1_model.inverse(cond_inp_M1, mask_M1_truth).detach().numpy()
                M1_samp = np.maximum(M1_samp, 0)
                M1_truth = torch.from_numpy(M1_samp).float().cuda()
            else:
                M1_truth = x_M1
        else:
            M1_truth = x_M1
            Nhalos_truth = x_Ntot

        logP_Mdiff = torch.zeros(1, device='cuda')
        if train_Mdiff:
            cond_inp_Mdiff = torch.cat([Nhalos_truth, M1_truth, cond_out], dim=1)
            if self.sep_Mdiff_cond:
                cond_inp_Mdiff = self.cond_Mdiff_layer(cond_inp_Mdiff)
            # m, _ = x_Mdiff.shape
            # log_det = torch.zeros(m, device='cuda')
            # for flow in self.flows_Mdiff:
            #     x_Mdiff, ld = flow.forward(x_Mdiff, cond_inp_Mdiff, mask_Mdiff_truth)
            #     log_det += ld
            # z = x_Mdiff

            # prior_logprob_all = torch.zeros(x_Mdiff.shape, device='cuda')
            # for jd in range(self.ndim - 1):
            #     prior_logprob_all[:, jd] = self.priors_all[jd].log_prob(z[:, jd])
            # prior_logprob_masked = prior_logprob_all * mask_Mdiff_truth
            # prior_logprob = torch.sum(prior_logprob_masked, axis=1)
            # logP_Mdiff = (torch.mean(prior_logprob + log_det))
            logP_Mdiff = self.Mdiff_model.forward(x_Mdiff, cond_inp_Mdiff, mask_Mdiff_truth)
        # print(logP_Ntot.shape, logP_M1.shape, logP_Mdiff.shape)
        loss = torch.mean(-logP_Ntot - logP_M1 - logP_Mdiff)

        return loss

    def inverse(
        self,
        cond_x=None,
        cond_x_nsh=None,
        use_truth_Nhalo=False,
        use_truth_M1=False,
        use_truth_Mdiff=False,
        mask_Mdiff_truth=None,
        mask_M1_truth=None,
        Nhalos_truth=None,
        M1_truth=None,
        train_Ntot=False,
        train_M1=False,
        train_Mdiff=False,
        ):
        cond_out = self.conv_layers(cond_x)
        cond_out = torch.cat((cond_out, cond_x_nsh), dim=1)
        if self.sep_Ntot_cond:
            cond_out_Ntot = self.cond_Ntot_layer(cond_out)
        else:
            cond_out_Ntot = cond_out

        if train_Ntot:
            Ntot_samp_tensor = self.Ntot_model.inverse(cond_out_Ntot)
            Ntot_samp = np.maximum(np.round(Ntot_samp_tensor.cpu().detach().numpy()) - 1, 0).astype(int)
        else:
            # Ntot_samp = torch.Tensor(Nhalos_truth)
            Ntot_samp = Nhalos_truth.cpu().detach().numpy()

        # nvox_batch = 64 // 8
        nvox_batch = self.nout // self.nbatch
        Ntot_samp_rs = Ntot_samp.reshape(-1, nvox_batch**3)
        # print(cond_out_Ntot.shape, Ntot_samp.shape, Ntot_samp_rs.shape)

        # Ntot_samp = np.maximum(np.round(self.Ntot_model.inverse(cond_out_Ntot).detach().numpy() - 1), 0).astype(int)
        nsim, nvox = Ntot_samp_rs.shape[0], Ntot_samp_rs.shape[1]
        mask_samp_all = np.zeros((nsim, nvox, self.ndim))
        idx = np.arange(self.ndim)[None, None, :]
        mask_samp_all[np.arange(nsim)[:, None, None],
                      np.arange(nvox)[None, :, None], idx] = (idx < Ntot_samp_rs[..., None])

        Ntot_samp_diff = Ntot_samp_rs - 1
        Ntot_samp_diff[Ntot_samp_diff < 0] = 0
        mask_samp_M_diff = np.zeros((nsim, nvox, self.ndim - 1))
        idx = np.arange(self.ndim - 1)[None, None, :]
        mask_samp_M_diff[np.arange(nsim)[:, None, None],
                         np.arange(nvox)[None, :, None], idx] = (idx < Ntot_samp_diff[..., None])

        mask_samp_M1 = mask_samp_all[:, :, 0]

        mask_samp_M_diff = mask_samp_M_diff.reshape(nsim * nvox, self.ndim - 1)
        mask_samp_M1 = mask_samp_M1.reshape(nsim * nvox, 1)

        if use_truth_M1:
            mask_tensor_M1_samp = (mask_M1_truth)
            mask_tensor_M1_samp = mask_tensor_M1_samp.float().cuda()

        else:
            # mask_tensor_M1_samp = torch.Tensor(np.array([mask_samp_all[:, 0]]).T)
            mask_tensor_M1_samp = torch.from_numpy(mask_samp_M1)
            mask_tensor_M1_samp = mask_tensor_M1_samp.float().cuda()

        if use_truth_Mdiff:
            mask_tensor_Mdiff_samp = (mask_Mdiff_truth)
        else:
            # mask_tensor_Mdiff_samp = torch.Tensor(np.copy(mask_samp))
            mask_tensor_Mdiff_samp = torch.from_numpy(mask_samp_M_diff)
            mask_tensor_Mdiff_samp = mask_tensor_Mdiff_samp.float().cuda()

        if use_truth_Nhalo:
            Nhalo_conditional = Nhalos_truth
        else:
            if train_Ntot:
                Nhalo_conditional = torch.Tensor(np.array([Ntot_samp]).T)
                Nhalo_conditional = Nhalo_conditional.float().cuda()
            else:
                raise ValueError('Must use truth Nhalo if not training Ntot')

        cond_inp_M1 = torch.cat([Nhalo_conditional, cond_out], dim=1)
        if self.sep_M1_cond:
            cond_inp_M1 = self.cond_M1_layer(cond_inp_M1)

        if train_M1:
            M1_samp, _ = self.M1_model.inverse(cond_inp_M1, mask_tensor_M1_samp)
        else:
            M1_samp = None

        if use_truth_M1:
            M1_conditional = M1_truth
        else:
            if train_M1:
                M1_conditional = torch.unsqueeze(M1_samp, 0).T
            else:
                raise ValueError('Must use truth M1 if not training M1')

        if train_Mdiff:
            cond_inp_Mdiff = torch.cat([Nhalo_conditional, M1_conditional, cond_out], dim=1)
            if self.sep_Mdiff_cond:
                cond_inp_Mdiff = self.cond_Mdiff_layer(cond_inp_Mdiff)

            # nsamp = cond_out.shape[0]
            # z = torch.zeros(mask_tensor_Mdiff_samp.shape)
            # for jd in range(z.shape[1]):
            #     z[:, jd] = self.priors_all[jd].sample((nsamp,))[:, 0]

            # m, _ = z.shape
            # log_det = torch.zeros(m)
            # log_det = log_det.cuda()
            # for flow in self.flows_Mdiff[::-1]:
            #     z, ld = flow.inverse(z, cond_inp_Mdiff, mask_tensor_Mdiff_samp)
            #     log_det += ld
            # x = z
            # M_diff_samp = x
            M_diff_samp, _ = self.Mdiff_model.inverse(cond_inp_Mdiff, mask_tensor_Mdiff_samp)
        else:
            M_diff_samp = None
        return Ntot_samp, M1_samp, M_diff_samp, mask_tensor_M1_samp, mask_tensor_Mdiff_samp
