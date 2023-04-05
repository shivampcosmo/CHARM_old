"""
This script processes the quijote density fields.
It will split the input simulation into sub-cubes and save them.
The sub-cube sizes account for the padding required for the CNN.
"""

import numpy as np
import os
import sys
import readgadget
import MAS_library as MASL


# This routine computes the density field and save results to file
def compute_df(snapshot, ptypes, grid, fout):
    if os.path.exists(fout):
        print('File already exists: ' + fout)
        return
    else:
        df = MASL.density_field_gadget(snapshot, ptypes, grid, MAS='CIC', do_RSD=False, axis=0, verbose=True)
        df = df / np.mean(df, dtype=np.float64) - 1.0
        np.save(fout, df)
        return


def save_subvol(
        ji,
        n_inp,
        n_filter,
        snapnum,
        n_batch=8,
        n_cnn=7,
        sim_dir='/pscratch/sd/s/spandey/quijote/Snapshot_fid/Snapshot_fid_density'
    ):
    # this is the number of sub-cubes into which the input simulation is split, but keeping in mind n_out

    n_dim_red = (n_filter - 1) // 2

    n_pad = n_dim_red * n_cnn
    z = {4: 0, 3: 0.5, 2: 1, 1: 2, 0: 3, -1: 127}[snapnum]

    # save the sub-cubes
    save_suffix = '_nbatch=' + str(n_batch) + '_nfilter=' + str(n_filter) + '_ncnn=' + str(n_cnn)
    save_fname = sim_dir + '/' + str(ji) + '/df_m_' + str(n_inp) + save_suffix + '_CIC_z=' + str(z) + '_subvol.npy'

    if os.path.exists(save_fname):
        print('File already exists: ' + save_fname)
        return
    else:
        # load the input simulation of size n_inp^3
        # df = np.load(sim_dir + '/' + str(ji) + '/df_m_' + str(n_inp) + '_CIC_z=' + str(z) + '.npy')
        df = np.load(sim_dir + '/' + str(ji) + '/df_m_' + str(n_inp) + '_z=' + str(z) + '.npy')

        # pad the simulation circularly with n_pad on all sides
        df_pad = np.pad(df, n_pad, 'wrap')

        # now df_pad has size (n_inp + n_pad)^3

        # we want to split the df_pad into n_batch^3 sub-cubes, but centered on the original df simulation box
        xstart, ystart, zstart = n_pad, n_pad, n_pad
        subvol_size = n_inp // n_batch + 2 * n_pad
        nsubvol = n_batch**3
        save_subvol = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size))
        jc = 0
        from tqdm import tqdm
        for jx in tqdm(range(n_batch)):
            for jy in range(n_batch):
                for jz in range(n_batch):
                    # get the sub-cube
                    df_sub = df_pad[xstart + jx * n_inp // n_batch - n_pad:xstart + (jx + 1) * n_inp // n_batch + n_pad,
                                    ystart + jy * n_inp // n_batch - n_pad:ystart + (jy + 1) * n_inp // n_batch + n_pad,
                                    zstart + jz * n_inp // n_batch - n_pad:zstart + (jz + 1) * n_inp // n_batch + n_pad]
                    # save the sub-cube
                    save_subvol[jc, ...] = df_sub
                    jc += 1

        # save the sub-cubes
        np.save(save_fname, save_subvol)
        return


if __name__ == '__main__':
    # this is the input box size
    n_inp_all = [64, 128, 256, 512]
    # n_inp_all = [512, 1024]
    # # this is the size of the box at which analysis is done (i.e. the halo catalog is at this resolution)
    # n_out = 64
    # this is the filter size of the CNN. Note that this is for the n_inp box size
    n_filter_all = [3]
    # n_filter_all = [11, 23]
    # this is the number by which the dimension of the input is reduced once a cnn operation happens in pytorch
    # see https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html

    ji_all = [0, 1]

    snapnum_all = [-1, 0, 1, 2, 3, 4]
    # snapnum_all = [-1]

    root = '/pscratch/sd/s/spandey/quijote/Snapshot_fid'
    root_out = '/pscratch/sd/s/spandey/quijote/Snapshot_fid/Snapshot_fid_density'
    ptypes = [1]

    for ji in ji_all:
        for n_inp in n_inp_all:
            for snapnum in snapnum_all:
                # create output folder if it does not exists
                folder_out = '%s/%d' % (root_out, ji)
                if not (os.path.exists(folder_out)):
                    os.system('mkdir %s' % folder_out)

                z = {4: 0, 3: 0.5, 2: 1, 1: 2, 0: 3, -1: 127}[snapnum]
                # find name of output file
                fout = '%s/df_m_%d_z=%s.npy' % (folder_out, n_inp, z)
                # if os.path.exists(fout):  continue

                # compute the density field and save it to file
                if z == 127:
                    snapshot = '%s/%d/ICs/ics' % (root, ji)
                else:
                    snapshot = '%s/%d/snapdir_%03d/snap_%03d' % (root, ji, snapnum, snapnum)
                print(snapshot)
                compute_df(snapshot, ptypes, n_inp, fout)

                for n_filter in n_filter_all:
                    save_subvol(ji, n_inp, n_filter, snapnum, n_cnn=0, sim_dir=root_out)
