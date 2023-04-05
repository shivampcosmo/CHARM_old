"""
This file processes the quijote catalogs.
Particularly, it reads the halo positions and masses. 
Then it will find the nearest cell to each halo and save the halo masses in that cell as an array.
Finally it will split this voxelized catalog into sub-cubes and save them.
"""

import sys, os
from tqdm import tqdm
import readfof
import numpy as np
import pickle as pk
from sklearn.neighbors import KDTree


def save_Mhalos_density_cells(
        ji,
        nside,
        snapnum=4,
        n_batch=8,
        snap_dir_base='/pscratch/sd/s/spandey/quijote/Halos/fiducial',
        Mmin=1e13,
        sdir='/pscratch/sd/s/spandey/quijote/Snapshot_fid/Snapshot_fid_density'
    ):
    savefname = sdir + '/' + str(ji) + '/halo_data_dict_' + str(nside) + '.pk'
    if os.path.exists(savefname):
        print('File already exists: ' + savefname)
        return
    else:
        # create the meshgrid
        zv = 0.0
        av = 1. / (1 + zv)
        xall = (np.linspace(0, 1000, nside + 1))
        xarray = av * 0.5 * (xall[1:] + xall[:-1])
        yarray = np.copy(xarray)
        zarray = np.copy(xarray)
        x_cy, y_cy, z_cy = np.meshgrid(xarray, yarray, zarray, indexing='ij')

        # load the corresponding halo catalogue
        snapdir = snap_dir_base + '/' + str(ji)  #folder hosting the catalogue

        # determine the redshift of the catalogue
        # z_dict = {4: 0.0, 3: 0.5, 2: 1.0, 1: 2.0, 0: 3.0}
        # redshift = z_dict[snapnum]

        # read the halo catalogue
        FoF = readfof.FoF_catalog(snapdir, snapnum, long_ids=False, swap=False, SFR=False, read_IDs=False)

        # get the properties of the halos
        pos_h = FoF.GroupPos / 1e3  #Halo positions in Mpc/h
        mass = FoF.GroupMass * 1e10  #Halo masses in Msun/h

        # select halos with mass > 1e13 Msun/h
        indsel = np.where((mass > Mmin))[0]
        # pos_x, pos_y, pos_z = pos_h[indsel, 0], pos_h[indsel, 1], pos_h[indsel, 2]
        lgMass_sel = np.log10(mass)[indsel]

        # find the nearest cell to each halo
        x_cy_flatten, y_cy_flatten, z_cy_flatten = x_cy.flatten(), y_cy.flatten(), z_cy.flatten()
        pcloud = np.vstack((x_cy_flatten, y_cy_flatten, z_cy_flatten)).T
        tree = KDTree(pcloud)
        _, ind = tree.query(pos_h, 1)

        # sort the indices
        ind_to_sort = np.argsort(ind[:, 0])
        M_sorted = lgMass_sel[ind_to_sort]
        ind_sorted = ind[ind_to_sort]
        if nside == 64:
            nMax_h = 30  # maximum number of halos expected in a cell
        elif nside == 128:
            nMax_h = 10
        elif nside == 256:
            nMax_h = 5
        elif nside == 512:
            nMax_h = 3
        elif nside == 1024:
            nMax_h = 2
        else:
            print('nside not supported')
            sys.exit()

        # count the number of halos in each cell
        bin_c, _ = np.histogram(ind_sorted[:, 0], np.arange(nside**3 + 1))
        k = 0
        M_halos = np.zeros((nside**3, nMax_h))
        Nhalos = np.zeros(nside**3)
        # save the halo masses in each cell
        for ji in (range(nside**3)):
            M_halos[ji, :bin_c[ji]] = M_sorted[k:bin_c[ji] + k]
            Nhalos[ji] = bin_c[ji]
            k += bin_c[ji]

        # reshape Nhalo and M_halo into 3D arrays
        Nhalos = Nhalos.reshape(nside, nside, nside)
        M_halos = M_halos.reshape(nside, nside, nside, nMax_h)

        # now split it into nbatches each side
        subvol_size = nside // n_batch
        nsubvol = n_batch**3
        save_subvol_Nhalo = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size))
        save_subvol_Mhalo = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size, nMax_h))
        jc = 0
        from tqdm import tqdm
        for jx in tqdm(range(n_batch)):
            for jy in range(n_batch):
                for jz in range(n_batch):
                    # get the sub-cube
                    save_subvol_Nhalo[jc] = Nhalos[jx * subvol_size:(jx + 1) * subvol_size,
                                                   jy * subvol_size:(jy + 1) * subvol_size,
                                                   jz * subvol_size:(jz + 1) * subvol_size]
                    save_subvol_Mhalo[jc] = M_halos[jx * subvol_size:(jx + 1) * subvol_size,
                                                    jy * subvol_size:(jy + 1) * subvol_size,
                                                    jz * subvol_size:(jz + 1) * subvol_size]
                    jc += 1

        saved = {
            'N_halos': save_subvol_Nhalo,
            'M_halos': save_subvol_Mhalo,
            'N_halos_combined': Nhalos,
            'M_halos_combined': M_halos
            }
        pk.dump(saved, open(savefname, 'wb'))


if __name__ == '__main__':
    # this is the input box size
    n_inp_all = [64, 128, 256]
    # these are the simulation numbers
    ji_all = [0, 1]

    for ji in ji_all:
        for nside in n_inp_all:
            save_Mhalos_density_cells(ji, nside)
