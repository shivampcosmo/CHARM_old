import sys, os
import pickle as pk
import numpy as np
from scipy.ndimage import gaussian_filter


# def load_density_halo_data(
#         ji,
#         nside_d,
#         nbatch,
#         nfilter,
#         ncnn,
#         z_all,
#         nside_h,
#         sdir='/pscratch/sd/s/spandey/quijote/Snapshot_fid/Snapshot_fid_density'
#     ):
#     # load the density data
#     df_d0 = np.load(
#         sdir + '/' + str(ji) + '/df_m_' + str(nside_d) + '_nbatch=' + str(nbatch) + '_nfilter=' + str(nfilter) +
#         '_ncnn=' + str(ncnn) + '_CIC_z=0_subvol.npy'
#         )
#     df_d_all = np.zeros((df_d0.shape[0], len(z_all), df_d0.shape[1], df_d0.shape[2], df_d0.shape[3]))
#     for iz, z in enumerate(z_all):
#         df_d_all[:, iz, ...] = np.load(
#             sdir + '/' + str(ji) + '/df_m_' + str(nside_d) + '_nbatch=' + str(nbatch) + '_nfilter=' + str(nfilter) +
#             '_ncnn=' + str(ncnn) + '_CIC_z=' + str(z) + '_subvol.npy'
#             )

#     df_d_all = np.log(1 + df_d_all + 1e-5)

#     # this is density at the output nside of CNN, which is same as halo nside
#     df_d0 = np.load(
#         sdir + '/' + str(ji) + '/df_m_' + str(nside_h) + '_nbatch=' + str(nbatch) + '_nfilter=' + str(nfilter) +
#         '_ncnn=' + str(0) + '_CIC_z=0_subvol.npy'
#         )
#     df_d_all_nsh = np.zeros((df_d0.shape[0], len(z_all), df_d0.shape[1], df_d0.shape[2], df_d0.shape[3]))
#     for iz, z in enumerate(z_all):
#         df_d_all_nsh[:, iz, ...] = np.load(
#             sdir + '/' + str(ji) + '/df_m_' + str(nside_h) + '_nbatch=' + str(nbatch) + '_nfilter=' + str(nfilter) +
#             '_ncnn=' + str(0) + '_CIC_z=' + str(z) + '_subvol.npy'
#             )

#     df_d_all_nsh = np.log(1 + df_d_all_nsh + 1e-5)

#     # load the halo data
#     fname = sdir + '/' + str(ji) + '/halo_data_dict_' + str(nside_h) + '.pk'
#     df_h = pk.load(open(fname, 'rb'))
#     # This has information on the halo mass for all the halos in the voxel
#     df_Mh_all = df_h['M_halos']
#     # This has information on the number of halos in the voxel
#     df_Nh = df_h['N_halos']
#     return df_d_all, df_d_all_nsh, df_Mh_all, df_Nh



def load_density_halo_data_NGP(
        ji_array,
        nside_d,
        nbatch,
        nfilter,
        ncnn,
        z_all,
        nside_h,
        sdir='/pscratch/sd/s/spandey/quijote/data_NGP_self',
        stype='cic',
        mass_type='fof',
        lgMmincutstr = '5e13',
        Mmin=14.0,
        subsel_highM1=False,
        nsubsel=512,
        ind_subsel=None,
        is_HR = False
    ):
    # load the halo data
    # fname = sdir + '/' + str(ji) + '/halo_data_dict_' + str(nside_h) + '.pk'
    for ji in range(len(ji_array)):
        jsim = ji_array[ji]
        if 'fastpm' in sdir:
            fname = sdir + '/' + str(jsim) + '/halos_subvol_res_' + str(nside_h) + '_z=' + str(0) + '.pk'
        else:
            if is_HR:
                fname = sdir + '/' + str(jsim) + '/halos_HR_' + mass_type + '_lgMmincut_' + lgMmincutstr + '_subvol_res_' + str(nside_h) + '_z=' + str(0) + '.pk'
            else:
                fname = sdir + '/' + str(jsim) + '/halos_' + mass_type + '_lgMmincut_' + lgMmincutstr + '_subvol_res_' + str(nside_h) + '_z=' + str(0) + '.pk'

        # if 'fof' in mass_type:
        #     # try:
        #     #     fname = sdir + '/' + str(jsim) + '/halos_subvol_res_' + str(nside_h) + '_z=' + str(0) + '.pk'
        #     # except:
        #     if is_HR:
        #         fname = sdir + '/' + str(jsim) + '/halos_' + mass_type + '_lgMmincut_' + lgMmincutstr + '_subvol_res_' + str(nside_h) + '_z=' + str(0) + '.pk'
        #     else:
        #         fname = sdir + '/' + str(jsim) + '/halos_HR_' + mass_type + '_lgMmincut_' + lgMmincutstr + '_subvol_res_' + str(nside_h) + '_z=' + str(0) + '.pk'
        # elif 'rockstar' in mass_type:
        #     fname = sdir + '/' + str(jsim) + '/halos_' + mass_type + '_lgMmincut_' + lgMmincutstr + '_subvol_res_' + str(nside_h) + '_z=' + str(0) + '.pk'
        # else:
        #     raise ValueError('mass_type not recognized')
        # print(fname)

        df_h = pk.load(open(fname, 'rb'))
        if ji == 0:
            # This has information on the halo mass for all the halos in the voxel
            df_Mh_all = df_h['M_halos']
            # This has information on the number of halos in the voxel
            df_Nh_all = df_h['N_halos']
        else:
            # stack along the first axis:
            df_Mh_all = np.concatenate((df_Mh_all, df_h['M_halos']), axis=0)
            df_Nh_all = np.concatenate((df_Nh_all, df_h['N_halos']), axis=0)

    # jsim = ji_array[0]
    # load the density data
    if is_HR and 'fastpm' not in sdir:
        df_load = pk.load(open(
            sdir + '/' + str(ji_array[0]) + '/density_HR_subvol_m_res_' + str(nside_d) + '_z=' + str(0) + '_nbatch_' + str(nbatch) + '_nfilter_' + str(nfilter) + '_ncnn_' + str(ncnn) + '.pk', 'rb')
            )

    else:
        df_load = pk.load(open(
            sdir + '/' + str(ji_array[0]) + '/density_subvol_m_res_' + str(nside_d) + '_z=' + str(0) + '_nbatch_' + str(nbatch) + '_nfilter_' + str(nfilter) + '_ncnn_' + str(ncnn) + '.pk', 'rb')
            )
    if stype == 'cic':
        df_d0 = df_load['density_cic_pad']
    if stype == 'uniform_cic':
        df_d0 = df_load['density_uniform_cic_pad']
    if stype == 'ngp':
        df_d0 = df_load['density_ngp_pad']
    ji0_shape = df_d0.shape[0]
    df_d_all = np.zeros((len(ji_array)*ji0_shape, len(z_all), df_d0.shape[1], df_d0.shape[2], df_d0.shape[3]))
    for ji in range(len(ji_array)):
        jsim = ji_array[ji]
        for iz, z in enumerate(z_all):
            # if z is a type of float, then it is a redshift:
            if isinstance(z, float) or isinstance(z, int):
                if is_HR and 'fastpm' not in sdir:
                    df_load = pk.load(open(
                        sdir + '/' + str(jsim) + '/density_HR_subvol_m_res_' + str(nside_d) + '_z=' + str(z) + '_nbatch_' + str(nbatch) + '_nfilter_' + str(nfilter) + '_ncnn_' + str(ncnn) + '.pk', 'rb')
                        )
                else:
                    df_load = pk.load(open(
                        sdir + '/' + str(jsim) + '/density_subvol_m_res_' + str(nside_d) + '_z=' + str(z) + '_nbatch_' + str(nbatch) + '_nfilter_' + str(nfilter) + '_ncnn_' + str(ncnn) + '.pk', 'rb')
                        )

                if stype == 'cic':
                    df_d_all[ji*ji0_shape:(ji+1)*ji0_shape, iz, ...] = np.log(1 + df_load['density_cic_pad'] + 1e-10)
                if stype == 'uniform_cic':
                    df_d_all[ji*ji0_shape:(ji+1)*ji0_shape, iz, ...] = np.log(1 + df_load['density_uniform_cic_pad'] + 1e-10)
                if stype == 'ngp':
                    df_d_all[ji*ji0_shape:(ji+1)*ji0_shape, iz, ...] = np.log(1 + df_load['density_ngp_pad'] + 1e-10)
            # if z is a string, then its type is 'z_REDSHIFT_diff_sig_VALUE', where VALUE is a float. extract that REDSHIFT and VALUE
            elif isinstance(z, str):
                if z[0] == 'z':
                    z_REDSHIFT_diff_sig_VALUE = z
                    z_REDSHIFT = float(z_REDSHIFT_diff_sig_VALUE.split('_')[1])
                    if z_REDSHIFT == 0.0:
                        z_REDSHIFT = 0
                    VALUE = float(z_REDSHIFT_diff_sig_VALUE.split('_')[4])
                    if is_HR:
                        df_load = pk.load(open(
                            sdir + '/' + str(jsim) + '/density_HR_subvol_m_res_' + str(nside_d) + '_z=' + str(z_REDSHIFT) + '_nbatch_' + str(nbatch) + '_nfilter_' + str(nfilter) + '_ncnn_' + str(ncnn) + '.pk', 'rb')
                            )
                    else:
                        df_load = pk.load(open(
                            sdir + '/' + str(jsim) + '/density_subvol_m_res_' + str(nside_d) + '_z=' + str(z_REDSHIFT) + '_nbatch_' + str(nbatch) + '_nfilter_' + str(nfilter) + '_ncnn_' + str(ncnn) + '.pk', 'rb')
                            )

                    if stype == 'cic':
                        density_unsmoothed = df_load['density_cic_pad']
                    if stype == 'uniform_cic':
                        density_unsmoothed = df_load['density_uniform_cic_pad']
                    if stype == 'ngp':
                        density_unsmoothed = df_load['density_ngp_pad']
                    # now smooth it with a gaussian filter of width VALUE
                    density_smoothed = gaussian_filter(density_unsmoothed, sigma=VALUE, axes=[1,2,3])
                    # save the difference between smoothed and unsmoothed density
                    df_d_all[ji*ji0_shape:(ji+1)*ji0_shape, iz, ...] = density_smoothed - density_unsmoothed
                
                elif z[0] == 'M':
                    df_h = pk.load(open(
                        sdir + '/' + str(jsim) + '/halos_subvol_m_res_' + str(nside_d) + '_z=' + str(0) + '_nbatch_' + str(nbatch) + '_nfilter_' + str(nfilter) + '_ncnn_' + str(ncnn) + '.pk', 'rb')
                        )

                    M_id = z[1]
                    Mcond_iz = df_h['M' + str(M_id) + '_halos_pad']
                    Mcond_iz = (Mcond_iz - Mmin)/Mmin
                    df_d_all[ji*ji0_shape:(ji+1)*ji0_shape, iz, ...] = Mcond_iz



    # df_d_all = np.log(1 + df_d_all + 1e-5)

    # this is density at the output nside of CNN, which is same as halo nside
    # df_d0 = np.load(
    #     sdir + '/' + str(ji) + '/df_m_' + str(nside_h) + '_nbatch=' + str(nbatch) + '_nfilter=' + str(nfilter) +
    #     '_ncnn=' + str(0) + '_CIC_z=0_subvol.npy'
    #     )

    # ji = ji_array[0]
    if is_HR and 'fastpm' not in sdir:
        df_load = pk.load(open(
            sdir + '/' + str(ji_array[0]) + '/density_HR_subvol_m_res_' + str(nside_h) + '_z=' + str(0) + '_nbatch_' + str(nbatch) + '_nfilter_' + str(nfilter) + '_ncnn_' + str(0) + '.pk', 'rb')
            )
    else:
        df_load = pk.load(open(
            sdir + '/' + str(ji_array[0]) + '/density_subvol_m_res_' + str(nside_h) + '_z=' + str(0) + '_nbatch_' + str(nbatch) + '_nfilter_' + str(nfilter) + '_ncnn_' + str(0) + '.pk', 'rb')
            )

    if stype == 'cic':
        df_d0 = df_load['density_cic_pad']
    if stype == 'uniform_cic':
        df_d0 = df_load['density_uniform_cic_pad']
    if stype == 'ngp':
        df_d0 = df_load['density_ngp_pad']    
    ji0_shape = df_d0.shape[0]
    df_d_all_nsh = np.zeros((len(ji_array)*ji0_shape, len(z_all), df_d0.shape[1], df_d0.shape[2], df_d0.shape[3]))
    for ji in range(len(ji_array)):
        jsim = ji_array[ji]
        for iz, z in enumerate(z_all):
            if isinstance(z, float) or isinstance(z, int):
                if is_HR and 'fastpm' not in sdir:
                    df_load = pk.load(open(
                        sdir + '/' + str(jsim) + '/density_HR_subvol_m_res_' + str(nside_h) + '_z=' + str(z) + '_nbatch_' + str(nbatch) + '_nfilter_' + str(nfilter) + '_ncnn_' + str(0) + '.pk', 'rb')
                        )
                else:
                    df_load = pk.load(open(
                        sdir + '/' + str(jsim) + '/density_subvol_m_res_' + str(nside_h) + '_z=' + str(z) + '_nbatch_' + str(nbatch) + '_nfilter_' + str(nfilter) + '_ncnn_' + str(0) + '.pk', 'rb')
                        )

                if stype == 'cic':
                    df_d_all_nsh[ji*ji0_shape:(ji+1)*ji0_shape, iz, ...] = np.log(1 + df_load['density_cic_pad'] + 1e-10)
                if stype == 'uniform_cic':
                    df_d_all_nsh[ji*ji0_shape:(ji+1)*ji0_shape, iz, ...] = np.log(1 + df_load['density_uniform_cic_pad'] + 1e-10)
                if stype == 'ngp':
                    df_d_all_nsh[ji*ji0_shape:(ji+1)*ji0_shape, iz, ...] = np.log(1 + df_load['density_ngp_pad'] + 1e-10)
            elif isinstance(z, str):
                if z[0] == 'z':
                    z_REDSHIFT_diff_sig_VALUE = z
                    z_REDSHIFT = float(z_REDSHIFT_diff_sig_VALUE.split('_')[1])
                    if z_REDSHIFT == 0.0:
                        z_REDSHIFT = 0
                    VALUE = float(z_REDSHIFT_diff_sig_VALUE.split('_')[4])
                    if is_HR and 'fastpm' not in sdir:
                        df_load = pk.load(open(
                            sdir + '/' + str(jsim) + '/density_HR_subvol_m_res_' + str(nside_h) + '_z=' + str(z_REDSHIFT) + '_nbatch_' + str(nbatch) + '_nfilter_' + str(nfilter) + '_ncnn_' + str(0) + '.pk', 'rb')
                            )
                    else:
                        df_load = pk.load(open(
                            sdir + '/' + str(jsim) + '/density_subvol_m_res_' + str(nside_h) + '_z=' + str(z_REDSHIFT) + '_nbatch_' + str(nbatch) + '_nfilter_' + str(nfilter) + '_ncnn_' + str(0) + '.pk', 'rb')
                            )

                    if stype == 'cic':
                        density_unsmoothed = df_load['density_cic_pad']
                    if stype == 'uniform_cic':
                        density_unsmoothed = df_load['density_uniform_cic_pad']
                    if stype == 'ngp':
                        density_unsmoothed = df_load['density_ngp_pad']
                    # now smooth it with a gaussian filter of width VALUE
                    density_smoothed = gaussian_filter(density_unsmoothed, sigma=VALUE, axes=[1,2,3])
                    # save the difference between smoothed and unsmoothed density
                    df_d_all_nsh[ji*ji0_shape:(ji+1)*ji0_shape, iz, ...] = density_smoothed - density_unsmoothed
                elif z[0] == 'M':
                    df_h = pk.load(open(
                        sdir + '/' + str(jsim) + '/halos_subvol_m_res_' + str(nside_h) + '_z=' + str(0) + '_nbatch_' + str(nbatch) + '_nfilter_' + str(nfilter) + '_ncnn_' + str(0) + '.pk', 'rb')
                        )

                    M_id = z[1]
                    Mcond_iz = df_h['M' + str(M_id) + '_halos_pad']
                    Mcond_iz = (Mcond_iz - Mmin)/Mmin
                    df_d_all_nsh[ji*ji0_shape:(ji+1)*ji0_shape, iz, ...] = Mcond_iz



        # df_d_all_nsh[:, iz, ...] = np.load(
        #     sdir + '/' + str(ji) + '/df_m_' + str(nside_h) + '_nbatch=' + str(nbatch) + '_nfilter=' + str(nfilter) +
        #     '_ncnn=' + str(0) + '_CIC_z=' + str(z) + '_subvol.npy'
        #     )

    # df_d_all_nsh = np.log(1 + df_d_all_nsh + 1e-5)
    if subsel_highM1:
        if ind_subsel is None:
            M1_all = df_Mh_all[...,0]
            M1_max_all = np.max(M1_all, axis=(1,2,3))
            argsort_M1 = np.flip(np.argsort(M1_max_all))[:nsubsel]
            df_d_all = df_d_all[argsort_M1, ...]
            df_d_all_nsh = df_d_all_nsh[argsort_M1, ...]
            df_Mh_all = df_Mh_all[argsort_M1, ...]
            df_Nh_all = df_Nh_all[argsort_M1, ...]
            ind_subsel = argsort_M1
        else:
            df_d_all = df_d_all[ind_subsel, ...]
            df_d_all_nsh = df_d_all_nsh[ind_subsel, ...]
            df_Mh_all = df_Mh_all[ind_subsel, ...]
            df_Nh_all = df_Nh_all[ind_subsel, ...]
    else:
        ind_subsel = np.arange(df_d_all.shape[0])

    return df_d_all, df_d_all_nsh, df_Mh_all, df_Nh_all, ind_subsel


def prep_density_halo_cats(
        df_d_all, df_d_all_nsh, df_Mh_all, df_Nh, nsims=None, nstart=None, Mmin=13.1, Mmax=16.0, rescaleM_sub=0.0, Nmax=None, sigv=0.05
    ):
    if nstart is None:
        if nsims is None:
            nsims = df_d_all.shape[0]
        # We only need the first nsims
        df_Mh_all = df_Mh_all[:nsims, ...]
        df_Nh = df_Nh[:nsims, ...]
        df_d_all = df_d_all[:nsims, ...]
        df_d_all_nsh = df_d_all_nsh[:nsims, ...]
    else:
        nend = nstart + nsims
        df_Mh_all = df_Mh_all[nstart:nend, ...]
        df_Nh = df_Nh[nstart:nend, ...]
        df_d_all = df_d_all[nstart:nend, ...]
        df_d_all_nsh = df_d_all_nsh[nstart:nend, ...]
        # print(nstart, nend)
    # Now we reshape the number of halos into 2D array of shape number of sub-sim, nvoxels (per sub-sim)
    # Note that the number of sub-sim = nb**3
    N_halos_all = df_Nh.reshape((df_Nh.shape[0], df_Nh.shape[1] * df_Nh.shape[2] * df_Nh.shape[3]))
    # Do the same for the halo mass
    M_halos_all = df_Mh_all.reshape(
        (df_Mh_all.shape[0], df_Mh_all.shape[1] * df_Mh_all.shape[2] * df_Mh_all.shape[3], df_Mh_all.shape[4])
        )

    # Sort the halo mass in descending order
    M_halos_all_sort = np.flip(np.sort(M_halos_all, axis=-1), axis=-1)
    # Scale the halo masses to be between 0 and 1
    M_halos_all_sort_norm = rescaleM_sub + ((M_halos_all_sort - Mmin) / (Mmax - Mmin))
    # indices where M_halos_all_sort_norm is less than rescaleM_sub
    ind_rescaleM_sub = np.where(M_halos_all_sort_norm < rescaleM_sub)
    # set those values to rescaleM_sub
    M_halos_all_sort_norm[ind_rescaleM_sub] = rescaleM_sub

    # If the halo mass is negative, set it to some small value close to zero
    # ind_neg = np.where(M_halos_all_sort_norm < 0)
    # M_halos_all_sort_norm[ind_neg] = 1e-4

    # This creates a mask for the halo mass matrix. The mask is 1 in the last axis corresponding to number of halos in that voxel
    mask_all = np.zeros((N_halos_all.shape[0], N_halos_all.shape[1], M_halos_all_sort.shape[-1]))
    idx = np.arange(M_halos_all_sort.shape[-1])[None, None, :]
    mask_all[np.arange(N_halos_all.shape[0])[:, None, None],
             np.arange(N_halos_all.shape[1])[None, :, None], idx] = (idx < N_halos_all[..., None])

    # Also create a mask for mass difference. This is 1 if the halo more than one halo is present and 0 if it is not
    N_halos_diff = N_halos_all - 1
    N_halos_diff[N_halos_diff < 0] = 0
    mask_M_diff = np.zeros((N_halos_all.shape[0], N_halos_all.shape[1], M_halos_all_sort_norm.shape[-1] - 1))
    idx = np.arange(M_halos_all_sort_norm.shape[-1] - 1)[None, None, :]
    mask_M_diff[np.arange(N_halos_all.shape[0])[:, None, None],
                np.arange(N_halos_all.shape[1])[None, :, None], idx] = (idx < N_halos_diff[..., None])

    mask_M1 = mask_all[:, :, 0]

    # Heavist halo mass in each voxel
    M1_halos_all_norm = M_halos_all_sort_norm[..., 0]

    # Take the rest of the halo masses and create a diff array
    M_diff_halos_all_norm = M_halos_all_sort_norm[..., :-1] - M_halos_all_sort_norm[..., 1:]

    # Now we create a mask for the halo masses. This is needed for the loss function
    M_diff_halos_all_norm_masked = M_diff_halos_all_norm * mask_M_diff

    if Nmax is None:
        Nmax = int(np.amax(N_halos_all))
    mu_all = np.arange(Nmax + 1) + 1
    sig_all = sigv * np.ones_like(mu_all)
    Nhalo_train_mg = sig_all[0] * np.random.randn(N_halos_all.shape[0], N_halos_all.shape[1]) + (N_halos_all) + 1
    # Nhalo_train_mg_arr = np.array([Nhalo_train_mg]).T
    Nhalo_train_mg_arr = Nhalo_train_mg[..., np.newaxis]
    ngauss_Nhalo = Nmax + 1

    # final dict with all the required data to run the model
    return_dict = {}
    return_dict['df_d_all'] = np.array(df_d_all)
    return_dict['df_d_all_nsh'] = np.array(df_d_all_nsh)
    return_dict['M_halos_all_sort_norm'] = np.array(M_halos_all_sort_norm)
    return_dict['Mmin'] = Mmin
    return_dict['Mmax'] = Mmax
    return_dict['Nmax'] = Nmax
    return_dict['mask_M_diff'] = np.array(mask_M_diff)
    return_dict['mask_M1'] = np.array(mask_M1)

    return_dict['M1_halos_all_norm'] = np.array(M1_halos_all_norm)
    return_dict['M_diff_halos_all_norm_masked'] = np.array(M_diff_halos_all_norm_masked)
    return_dict['Nhalo_train_mg_arr'] = np.array(Nhalo_train_mg_arr)
    return_dict['N_halos_all'] = np.array(N_halos_all)
    return_dict['mu_all'] = np.array(mu_all)
    return_dict['sig_all'] = np.array(sig_all)
    return_dict['ngauss_Nhalo'] = ngauss_Nhalo

    return return_dict


def prep_density_halo_cats_batched(
        df_d_all_inp,
        df_d_all_nsh_inp,
        df_Mh_all_inp,
        df_Nh_inp,
        nsims=None,
        nbatches=1,
        Mmin=13.1,
        Mmax=16.0,
        rescaleM_sub=0.0,
        Nmax=None,
        sigv=0.05
    ):
    df_d_all_out = []
    df_d_all_nsh_out = []
    mask_M_diff_all, mask_M1_all = [], []
    M_halos_all_sort_norm_all, M1_halos_all_norm_all = [], []
    M_diff_halos_all_norm_masked_all = []
    Nhalo_train_mg_arr_all, N_halos_all_comb = [], []
    for jb in range(nbatches):
        nstart = jb * nsims
        nend = (jb + 1) * nsims
        df_Mh_all = df_Mh_all_inp[nstart:nend, ...]
        df_Nh = df_Nh_inp[nstart:nend, ...]
        df_d_all = df_d_all_inp[nstart:nend, ...]
        df_d_all_out.append(df_d_all)
        df_d_all_nsh = df_d_all_nsh_inp[nstart:nend, ...]
        df_d_all_nsh_out.append(df_d_all_nsh)
        # print(nstart, nend)
        # Now we reshape the number of halos into 2D array of shape number of sub-sim, nvoxels (per sub-sim)
        # Note that the number of sub-sim = nb**3
        N_halos_all = df_Nh.reshape((df_Nh.shape[0], df_Nh.shape[1] * df_Nh.shape[2] * df_Nh.shape[3]))
        N_halos_all_comb.append(N_halos_all)
        # Do the same for the halo mass
        M_halos_all = df_Mh_all.reshape(
            (df_Mh_all.shape[0], df_Mh_all.shape[1] * df_Mh_all.shape[2] * df_Mh_all.shape[3], df_Mh_all.shape[4])
            )

        # Sort the halo mass in descending order
        M_halos_all_sort = np.flip(np.sort(M_halos_all, axis=-1), axis=-1)
        # Scale the halo masses to be between 0 and 1
        M_halos_all_sort_norm = rescaleM_sub + (M_halos_all_sort - Mmin) / (Mmax - Mmin)
        # indices where M_halos_all_sort_norm is less than rescaleM_sub
        ind_rescaleM_sub = np.where(M_halos_all_sort_norm < rescaleM_sub)
        # set those values to rescaleM_sub
        M_halos_all_sort_norm[ind_rescaleM_sub] = rescaleM_sub

        M_halos_all_sort_norm_all.append(M_halos_all_sort_norm)

        # # If the halo mass is negative, set it to some small value close to zero
        # ind_neg = np.where(M_halos_all_sort_norm < 0)
        # M_halos_all_sort_norm[ind_neg] = 1e-4

        # This creates a mask for the halo mass matrix. The mask is 1 in the last axis corresponding to number of halos in that voxel
        mask_all = np.zeros((N_halos_all.shape[0], N_halos_all.shape[1], M_halos_all_sort.shape[-1]))
        idx = np.arange(M_halos_all_sort.shape[-1])[None, None, :]
        mask_all[np.arange(N_halos_all.shape[0])[:, None, None],
                 np.arange(N_halos_all.shape[1])[None, :, None], idx] = (idx < N_halos_all[..., None])

        # Also create a mask for mass difference. This is 1 if the halo more than one halo is present and 0 if it is not
        N_halos_diff = N_halos_all - 1
        N_halos_diff[N_halos_diff < 0] = 0
        mask_M_diff = np.zeros((N_halos_all.shape[0], N_halos_all.shape[1], M_halos_all_sort_norm.shape[-1] - 1))
        idx = np.arange(M_halos_all_sort_norm.shape[-1] - 1)[None, None, :]
        mask_M_diff[np.arange(N_halos_all.shape[0])[:, None, None],
                    np.arange(N_halos_all.shape[1])[None, :, None], idx] = (idx < N_halos_diff[..., None])

        mask_M1 = mask_all[:, :, 0]
        mask_M_diff_all.append(mask_M_diff)
        mask_M1_all.append(mask_M1)

        # Heavist halo mass in each voxel
        M1_halos_all_norm = M_halos_all_sort_norm[..., 0]
        M1_halos_all_norm_all.append(M1_halos_all_norm)

        # Take the rest of the halo masses and create a diff array
        M_diff_halos_all_norm = (M_halos_all_sort_norm[..., :-1] - M_halos_all_sort_norm[..., 1:])

        # Now we create a mask for the halo masses. This is needed for the loss function
        M_diff_halos_all_norm_masked = (M_diff_halos_all_norm * mask_M_diff)
        M_diff_halos_all_norm_masked_all.append(M_diff_halos_all_norm_masked)
        if Nmax is None:
            Nmax = int(np.amax(N_halos_all))
        mu_all = np.arange(Nmax + 1) + 1
        sig_all = sigv * np.ones_like(mu_all)
        Nhalo_train_mg = sig_all[0] * np.random.randn(N_halos_all.shape[0], N_halos_all.shape[1]) + (N_halos_all) + 1
        # Nhalo_train_mg_arr = np.array([Nhalo_train_mg]).T
        Nhalo_train_mg_arr = (Nhalo_train_mg[..., np.newaxis])
        Nhalo_train_mg_arr_all.append(Nhalo_train_mg_arr)
        ngauss_Nhalo = Nmax + 1

    # final dict with all the required data to run the model
    return_dict = {}
    return_dict['df_d_all'] = np.array(df_d_all_out)
    return_dict['df_d_all_nsh'] = np.array(df_d_all_nsh_out)
    return_dict['M_halos_all_sort_norm'] = np.array(M_halos_all_sort_norm_all)
    return_dict['Mmin'] = Mmin
    return_dict['Mmax'] = Mmax
    return_dict['Nmax'] = Nmax
    return_dict['mask_M_diff'] = np.array(mask_M_diff_all)
    return_dict['mask_M1'] = np.array(mask_M1_all)

    return_dict['M1_halos_all_norm'] = np.array(M1_halos_all_norm_all)
    return_dict['M_diff_halos_all_norm_masked'] = np.array(M_diff_halos_all_norm_masked_all)
    return_dict['Nhalo_train_mg_arr'] = np.array(Nhalo_train_mg_arr_all)
    return_dict['N_halos_all'] = np.array(N_halos_all_comb)
    return_dict['mu_all'] = np.array(mu_all)
    return_dict['sig_all'] = np.array(sig_all)
    return_dict['ngauss_Nhalo'] = ngauss_Nhalo

    return return_dict
