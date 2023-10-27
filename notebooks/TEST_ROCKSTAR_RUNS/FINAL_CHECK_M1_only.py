import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import pickle as pk
import numpy as np
import torch
dev = torch.device("cuda")
import torch.optim as optim
# from torch.distributions import MultivariateNormal
# from torch.distributions import Normal
root_dir = '/mnt/home/spandey/ceph/AR_NPE/'
os.chdir(root_dir)
# import colossus
import sys, os
# append the root_dir to the path
sys.path.append(root_dir)
from nf.combined_models import COMBINED_Model
from nf.all_models import *
from nf.utils_data_prep import *
# from tqdm import tqdm
# import pyyaml
from colossus.cosmology import cosmology
params = {'flat': True, 'H0': 67.11, 'Om0': 0.3175, 'Ob0': 0.049, 'sigma8': 0.834, 'ns': 0.9624}
cosmo = cosmology.setCosmology('myCosmo', **params)
# get halo mass function:
from colossus.lss import mass_function
from tqdm import tqdm
    
import yaml

import matplotlib
import matplotlib.pyplot as pl

# run_config_name = sys.argv[1]
run_config_name = 'run_M1only_128_condFPM_uniformcic_fof_lgMmin1e13_wL2norm_highknots.yaml'

with open("/mnt/home/spandey/ceph/AR_NPE/run_configs/" + run_config_name,"r") as file_object:
    config=yaml.load(file_object,Loader=yaml.SafeLoader)




config_sims = config['sim_settings']
ji_array = np.arange(int(config_sims['nsims']))
ns_d = config_sims['ns_d']
nb = config_sims['nb']
nax_d =  ns_d // nb
nf = config_sims['nf']
layers_types = config_sims['layers_types']
nc = 0
for jl in range(len(layers_types)):
    if layers_types[jl] == 'cnn':
        nc += 1
    elif layers_types[jl] == 'res':
        nc += 2
    else:
        raise ValueError("layer type not supported")

z_all = config_sims['z_all']
z_all_FP = config_sims['z_all_FP']
ns_h = config_sims['ns_h']
nax_h = ns_h // nb
cond_sim = config_sims['cond_sim']

nsims_per_batch = config_sims['nsims_per_batch']
nbatches_train = config_sims['nbatches_train']

mass_type = config_sims['mass_type']
lgMmin = config_sims['lgMmin']
lgMmax = config_sims['lgMmax']
stype = config_sims['stype']
rescale_sub = config_sims['rescale_sub']
lgMmincutstr = config_sims['lgMmincutstr']
subsel_highM1 = config_sims['subsel_highM1']
nsubsel = config_sims['nsubsel']
is_HR = config_sims['is_HR']

try:
    Nmax = config_sims['Nmax']
except:
    Nmax = 4

config_net = config['network_settings']
hidden_dim_MAF = config_net['hidden_dim_MAF']
learning_rate = config_net['learning_rate']
K_M1 = config_net['K_M1']
B_M1 = config_net['B_M1']
nflows_M1_NSF = config_net['nflows_M1_NSF']

K_Mdiff = config_net['K_Mdiff']
B_Mdiff = config_net['B_Mdiff']
nflows_Mdiff_NSF = config_net['nflows_Mdiff_NSF']

base_dist_Ntot = config_net['base_dist_Ntot']
if base_dist_Ntot == 'None':
    base_dist_Ntot = None
base_dist_M1 = config_net['base_dist_M1']
base_dist_Mdiff = config_net['base_dist_Mdiff']
ngauss_M1 = config_net['ngauss_M1']

changelr = config_net['changelr']
ksize = nf
nfeature_cnn = config_net['nfeature_cnn']
nout_cnn = 4 * nfeature_cnn
if cond_sim == 'fastpm':
    ninp = len(z_all_FP)
elif cond_sim == 'quijote':
    ninp = len(z_all)
else:
    raise ValueError("cond_sim not supported")

num_cond = nout_cnn + ninp

df_d_all_train, df_d_all_nsh_train, df_Mh_all_train, df_Nh_train, ind_subsel_train = load_density_halo_data_NGP(
    ji_array, ns_d, nb, nf, nc, z_all, ns_h,sdir='/mnt/home/spandey/ceph/Quijote/data_NGP_self',
    stype=stype, mass_type=mass_type, lgMmincutstr = lgMmincutstr, subsel_highM1=subsel_highM1, nsubsel=nsubsel
    )

# # Prepare the density and halo data
return_dict_train = prep_density_halo_cats_batched(
    df_d_all_train, df_d_all_nsh_train, df_Mh_all_train, df_Nh_train, nsims=nsims_per_batch,
    nbatches = nbatches_train, Mmin=lgMmin, Mmax=lgMmax, rescaleM_sub=rescale_sub, Nmax=Nmax
    )

if cond_sim == 'fastpm':
    df_d_all_train_FP, df_d_all_nsh_train_FP, df_Mh_all_train_FP, df_Nh_train_FP, ind_subsel_train_FP = load_density_halo_data_NGP(
        ji_array, ns_d, nb, nf, nc, z_all_FP, ns_h, stype=stype,sdir='/mnt/home/spandey/ceph/Quijote/data_NGP_self/fastpm', subsel_highM1=subsel_highM1, ind_subsel=ind_subsel_train
        )

    # # Prepare the density and halo data
    return_dict_train_FP = prep_density_halo_cats_batched(
        df_d_all_train_FP, df_d_all_nsh_train_FP, df_Mh_all_train_FP, df_Nh_train_FP, nsims=nsims_per_batch,
        nbatches = nbatches_train, Mmin=lgMmin, Mmax=lgMmax, rescaleM_sub=rescale_sub
        )
else:
    return_dict_train_FP = None





lgM_array = np.linspace(lgMmin, lgMmax, 1000)
M_array = 10**lgM_array
if '200c' in mass_type:
    hmf = mass_function.massFunction(M_array, 0, mdef = '200c', model = 'tinker08', q_out = 'dndlnM')
if 'vir' in mass_type:
    hmf = mass_function.massFunction(M_array, 0, mdef = 'vir', model = 'tinker08', q_out = 'dndlnM')    
if 'fof' in mass_type:
    hmf = mass_function.massFunction(M_array, 0, mdef = 'fof', model = 'bhattacharya11', q_out = 'dndlnM')
lgM_rescaled = rescale_sub + (lgM_array - lgMmin)/(lgMmax-lgMmin)

int_val = sp.integrate.simps(hmf, lgM_rescaled)
hmf_pdf = hmf/int_val
# define the cdf of the halo mass function
hmf_cdf = np.zeros_like(hmf_pdf)
for i in range(len(hmf_cdf)):
    hmf_cdf[i] = sp.integrate.simps(hmf_pdf[:i+1], lgM_rescaled[:i+1])

from torch.utils.data import DataLoader, Dataset


ndim_diff = return_dict_train['M_diff_halos_all_norm_masked'][0].shape[2]

if return_dict_train_FP is not None:
    cond_tensor = torch.Tensor(np.array(return_dict_train_FP['df_d_all']))
    cond_nsh = np.moveaxis(np.array(return_dict_train_FP['df_d_all_nsh']), 2, 5)
    cond_tensor_nsh = torch.Tensor((cond_nsh)).reshape(-1, nsims_per_batch * (nax_h ** 3), ninp)
else:
    cond_tensor = torch.Tensor(np.array(return_dict_train['df_d_all']))
    cond_nsh = np.moveaxis(np.array(return_dict_train['df_d_all_nsh']), 2, 5)
    cond_tensor_nsh = torch.Tensor((cond_nsh)).reshape(-1, nsims_per_batch * (nax_h ** 3), ninp)

mask_tensor_M1_train = torch.Tensor(np.array(return_dict_train['mask_M1'])).reshape(-1, nsims_per_batch * (nax_h**3))
mask_tensor_Mdiff_train = torch.Tensor((np.array(return_dict_train['mask_M_diff']))).reshape(-1, nsims_per_batch * (nax_h**3), ndim_diff)

X_M1 = torch.Tensor((np.array(return_dict_train['M1_halos_all_norm']))).reshape(-1, nsims_per_batch * (nax_h**3), 1)
X_Nhalo = torch.Tensor(np.array(return_dict_train['Nhalo_train_mg_arr'])).reshape(-1, nsims_per_batch * (nax_h**3), 1)
X_Mdiff = torch.Tensor(np.array(return_dict_train['M_diff_halos_all_norm_masked'])).reshape(-1, nsims_per_batch * (nax_h**3),ndim_diff)
Nhalos_truth_tensor = torch.Tensor(((np.array(return_dict_train['N_halos_all'])))).reshape(-1, nsims_per_batch * (nax_h**3), 1)

if return_dict_train_FP is not None:
    mask_tensor_M1_train_FP = torch.Tensor(np.array(return_dict_train_FP['mask_M1'])).reshape(-1, nsims_per_batch * (nax_h**3))
    X_M1_FP = torch.Tensor((np.array(return_dict_train_FP['M1_halos_all_norm']))).reshape(-1, nsims_per_batch * (nax_h**3), 1)
else:
    mask_tensor_M1_train_FP = None
    X_M1_FP = None

    
cond_tensor = cond_tensor.cuda(dev)
cond_tensor_nsh = cond_tensor_nsh.cuda(dev)
mask_tensor_M1_train = mask_tensor_M1_train.cuda(dev)
mask_tensor_Mdiff_train = mask_tensor_Mdiff_train.cuda(dev)
X_M1 = X_M1.cuda(dev)
X_Nhalo = X_Nhalo.cuda(dev)
X_Mdiff = X_Mdiff.cuda(dev)


Nhalos_truth_tensor = Nhalos_truth_tensor.cuda(dev)
if return_dict_train_FP is not None:
    mask_tensor_M1_train_FP = mask_tensor_M1_train_FP.cuda(dev)
    X_M1_FP = X_M1_FP.cuda(dev)
    
    


with open("/mnt/home/spandey/ceph/AR_NPE/run_configs/" + run_config_name,"r") as file_object:
    config=yaml.load(file_object,Loader=yaml.SafeLoader)

config_train = config['train_settings']
batch_size = config_train['batch_size_DL']
all_gpu = config_train['all_gpu']

try:
    L2norm_Ntothist = config_train['L2norm_Ntothist']
except:
    L2norm_Ntothist = False

try:
    L2norm_M1hist = config_train['L2norm_M1hist']
except:
    L2norm_M1hist = False

nflows_train = config_train['nflows_train']

save_bestfit_model_dir = '/mnt/home/spandey/ceph/AR_NPE/' + 'TRAIN_ROCKSTAR_FOF/FINALTEST_SUBSEL_M1only_plexp_ns_' + str(len(ji_array)) + \
                            '_cond_sim_' + cond_sim  \
                            + '_nc' + str(nc) + '_mass_' + mass_type + \
                            '_KM1_' + str(K_M1) + \
                            '_stype_' + stype + \
                            '_L2normNtothist_' + str(L2norm_Ntothist) + '_L2normM1hist_' + str(L2norm_M1hist)

print(save_bestfit_model_dir, os.path.exists(save_bestfit_model_dir))
# make directory if it doesn't exist
import os
if not os.path.exists(save_bestfit_model_dir):
    os.makedirs(save_bestfit_model_dir)


nepochs_Ntot_only = config_train['nepochs_Ntot_only']
nepochs_Ntot_M1_only = config_train['nepochs_Ntot_M1_only']
nepochs_all = config_train['nepochs_all']


# nepochs_array = [nepochs_Ntot_only, nepochs_Ntot_M1_only, nepochs_all]
# train_Ntot_array = [1, 1, 1]
# train_M1_array = [0, 1, 1 ]
# train_Mdiff_array = [0, 0, 1]

nepochs_array = [nepochs_Ntot_M1_only]
train_Ntot_array = [0]
train_M1_array = [1]
train_Mdiff_array = [0]

for jf in range(nflows_train):
# for jf in np.arange(7,16):    
    epoch_tot_counter = 0
    num_cond_Ntot = num_cond
    
    model_Ntot = SumGaussModel(
        hidden_dim=hidden_dim_MAF,
        num_cond=num_cond_Ntot,
        ngauss=return_dict_train['ngauss_Nhalo'],
        mu_all=return_dict_train['mu_all'],
        sig_all=return_dict_train['sig_all'],
        base_dist=base_dist_Ntot   
        )

    num_cond_M1 = num_cond + 1
    # if conditioned on fastpm we will also give the fastpm fof M1 halos and its mask as conditional
    if cond_sim == 'fastpm':
        num_cond_M1 += 2

    model_M1 = NSF_M1_CNNcond(
        K=K_M1,
        B=B_M1,
        hidden_dim=hidden_dim_MAF,
        num_cond=num_cond_M1,
        nflows=nflows_M1_NSF,
        base_dist=base_dist_M1,
        ngauss=ngauss_M1,
        lgM_rs_tointerp=lgM_rescaled,
        hmf_pdf_tointerp=hmf_pdf,
        hmf_cdf_tointerp=hmf_cdf    
        )

    # ndim_diff = return_dict_train['M_diff_halos_all_norm_masked'][0].shape[2]
    # num_cond_Mdiff = num_cond + 2
    # model_Mdiff = NSF_Mdiff_CNNcond(
    #     dim=ndim_diff,
    #     K=K_Mdiff,
    #     B=B_Mdiff,
    #     hidden_dim=hidden_dim_MAF,
    #     num_cond=num_cond_Mdiff,
    #     nflows=nflows_Mdiff_NSF,
    #     base_dist=base_dist_Mdiff,
    #     mu_pos=True
    #     )

    ndim = ndim_diff + 1
    model = COMBINED_Model(
        None,
        None,
        # None,
        model_M1,
        model_Ntot,
        ndim,
        ksize,
        ns_d,
        ns_h,
        nb,
        ninp,
        nfeature_cnn,
        nout_cnn,
        layers_types=layers_types,
        act='tanh',
        padding='valid',
        sep_Ntot_cond=True,
        sep_M1_cond=True,
        sep_Mdiff_cond=True,
        num_cond_Ntot = num_cond_Ntot,
        num_cond_M1 = num_cond_M1,
        num_cond_Mdiff = None
        )

    model.to(dev)

    print()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_all_it = []
    loss_min = 1e20
    epoch_tot_counter = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=300, verbose=True, cooldown=100, min_lr=1e-8)



    save_bestfit_model_name = save_bestfit_model_dir + '/flow_' + str(jf)
    for jn in (range(len(nepochs_array))):
        loss_min = 1e20
        torch.cuda.empty_cache()
        ninit = 0
        nepochs = nepochs_array[jn]
        if nepochs > 0:
            train_Ntot = train_Ntot_array[jn]
            train_M1 = train_M1_array[jn]
            train_Mdiff = train_Mdiff_array[jn]

            # if jn > 0:
            #     print('loading bestfit model')
            #     bestfit_model = (torch.load(save_bestfit_model_name))
            #     model.load_state_dict(bestfit_model['state_dict'])
            #     optimizer.load_state_dict(bestfit_model['optimizer'])
            #     scheduler.load_state_dict(bestfit_model['scheduler'])
            #     # loss_min = bestfit_model['loss_min']
            #     loss = bestfit_model['loss']
            #     lr = bestfit_model['lr']
            #     for g in optimizer.param_groups:
            #         g['lr'] = learning_rate

            for jt in (range(nepochs)):
                # for jd in range(len(dataloader)):
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                cond_tensor_jd, cond_tensor_nsh_jd, mask_tensor_M1_train_jd, mask_tensor_Mdiff_train_jd, X_M1_jd, \
                    X_Nhalo_jd, X_Mdiff_jd, Nhalos_truth_tensor_jd, mask_tensor_M1_train_FP_jd, X_M1_FP_jd = cond_tensor, cond_tensor_nsh, mask_tensor_M1_train, mask_tensor_Mdiff_train, X_M1, \
                            X_Nhalo, X_Mdiff, Nhalos_truth_tensor, mask_tensor_M1_train_FP, X_M1_FP

                if cond_sim == 'quijote':               
                    mask_tensor_M1_train_FP_jd = None
                    X_M1_FP_jd = None
                
                torch.cuda.empty_cache()
                # if 1-all_gpu:
                # cond_tensor_jd = cond_tensor_jd.cuda(dev)
                # cond_tensor_nsh_jd = cond_tensor_nsh_jd.cuda(dev)
                # mask_tensor_M1_train_jd = mask_tensor_M1_train_jd.cuda(dev)
                # mask_tensor_Mdiff_train_jd = mask_tensor_Mdiff_train_jd.cuda(dev)
                # X_M1_jd = X_M1_jd.cuda(dev)
                # X_Nhalo_jd = X_Nhalo_jd.cuda(dev)
                # X_Mdiff_jd = X_Mdiff_jd.cuda(dev)
                # Nhalos_truth_tensor_jd = Nhalos_truth_tensor_jd.cuda(dev)
                # if mask_tensor_M1_train_FP_jd is not None:
                #     mask_tensor_M1_train_FP_jd = mask_tensor_M1_train_FP_jd.cuda(dev)
                #     X_M1_FP_jd = X_M1_FP_jd.cuda(dev)
                # torch.cuda.empty_cache()
                                
                loss = model(
                    X_Mdiff_jd,
                    X_M1_jd,
                    X_Nhalo_jd,
                    cond_x=cond_tensor_jd,
                    cond_x_nsh=cond_tensor_nsh_jd,
                    mask_Mdiff_truth_all=mask_tensor_Mdiff_train_jd,
                    mask_M1_truth_all=mask_tensor_M1_train_jd,
                    Nhalos_truth_all=Nhalos_truth_tensor_jd,
                    use_Ntot_samples=False,
                    use_M1_samples=False,
                    train_Ntot=train_Ntot,
                    train_M1=train_M1,
                    train_Mdiff=train_Mdiff,
                    # x_Mdiff_FP=X_Mdiff_FP,
                    x_M1_FP=X_M1_FP_jd,
                    # x_Ntot_FP=X_Nhalo_FP,
                    # Nhalos_truth_all_FP=Nhalos_truth_tensor_FP,
                    # mask_Mdiff_truth_all_FP=mask_tensor_Mdiff_train_FP,
                    mask_M1_truth_all_FP=mask_tensor_M1_train_FP_jd,
                    L2norm_Ntothist=L2norm_Ntothist,
                    L2norm_M1hist=L2norm_M1hist        
                    )

                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                epoch_tot_counter += 1
                if (np.mod(jt, int(nepochs / 200)) == 0) or (jt == nepochs - 1):
                    if float(loss.cpu().detach().numpy()) < loss_min:
                        loss_min = float(loss.cpu().detach().numpy())
                        print('saving bf at:', ', with loss:', np.round(loss_min, 5), ', at epoch:', jt, 
                            'learning rate:', optimizer.param_groups[0]['lr'], ', train_Ntot:', train_Ntot, 
                            'train_M1:', train_M1, ', train_Mdiff:', train_Mdiff, ', epoch_tot_counter:', epoch_tot_counter)
                        lr=optimizer.param_groups[0]['lr']
                        # print(loss_min, lr)
                        state = {'loss_min': loss_min, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(), 'loss':loss, 'lr':lr, 'epoch_tot_counter':epoch_tot_counter}

                        torch.save(
                            state, save_bestfit_model_name
                            )




