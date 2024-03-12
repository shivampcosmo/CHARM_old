import numpy as np
import sys,os
import readgadget
import MAS_library as MASL
import pickle as pk
import readfof
import matplotlib

import matplotlib.pyplot as pl
pl.rc('text', usetex=True)
# Palatino
pl.rc('font', family='DejaVu Sans')
# %matplotlib inline
import cython
cimport numpy as np
cimport cython

# %load_ext Cython
from cpython cimport bool


cimport numpy as np

@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
cpdef void NGP_mass(np.float32_t[:,:] pos, np.float32_t[:] logM, np.float32_t[:,:,:,:] gridM, float BoxSize):

    cdef int axis,dims,coord,nMmax,jM
    cdef long i,particles
    cdef float inv_cell_size
    cdef int index[3]

    # find number of particles, the inverse of the cell size and dims
    particles = pos.shape[0];  coord = pos.shape[1];  dims = gridM.shape[0]; nMmax = gridM.shape[3]
    inv_cell_size = dims/BoxSize

    # when computing things in 2D, use the index[2]=0 plane
    for i in range(3):  index[i] = 0

    # do a loop over all particles
    for i in range(particles):
        for axis in range(coord):
            index[axis] = <int>(pos[i,axis]*inv_cell_size + 0.5)
            index[axis] = (index[axis]+dims)%dims
        for jM in range(nMmax):
            if gridM[index[0],index[1],index[2], jM] == 0:
                gridM[index[0],index[1],index[2], jM] = logM[i]
                break
            else:
                pass
        




# %pip install Pylians
from tqdm import tqdm
# root         = '/pscratch/sd/s/spandey/quijote/Snapshot_fid'
# root = '/mnt/home/spandey/ceph/Quijote/fiducial_HR_new/Snapshots'
root = '/mnt/home/fvillaescusa/ceph/Quijote/Snapshots/latin_hypercube_HR'
# snap_dir_base='/pscratch/sd/s/spandey/quijote/Halos/fiducial'
snap_dir_base = '/mnt/home/fvillaescusa/ceph/Quijote/Halos/FoF/latin_hypercube_HR'
# root_out     = '/pscratch/sd/s/spandey/quijote/data_NGP_self'
root_out = '/mnt/home/spandey/ceph/Quijote/data_NGP_self_LH/'
ptypes       = [1]
# snapnum      = 0
# grids         = [64, 128, 256, 512]
# grids         = [64, 128, 256]
grids         = [128]
# grids         = [128, 256]
BoxSize = 1000.0 #Mpc/h ; size of box
n_batch = 8
n_filter = 3
# n_cnn_all = [0,5,7]
# n_cnn_all = [8, 0]
# n_cnn_all = [0,4,6,8,12,16]
n_cnn_all = [0,4,8]
# n_sim_tot = 2
# n_sim_array = np.arange(0,10)
# n_sim_array = np.arange(0,20)
n_sim_array = np.arange(5,20)
# snap_num_array = [-1,0,1,2,3,4]
snap_num_array = [-1,4]
# snap_num_array = [-1]
# i = 0
# print(i)
for ji in tqdm(n_sim_array):
    # print('doing sim: ' + str(ji))
    for grid in grids:
        # print('doing res: ' + str(grid))
        for snapnum in snap_num_array:
            z = {4:0, 3:0.5, 2:1, 1:2, 0:3, -1: 127}[snapnum]

            # create output folder if it does not exists
            folder_out = '%s/%d'%(root_out,ji)
            if not(os.path.exists(folder_out)):
                os.system('mkdir %s'%folder_out)

            
            savefname_halos_subvol = '%s/halos_HR_subvol_res_%d_z=%s.pk'%(folder_out,grid,z)
            savefname_halos_full = '%s/halos_HR_full_res_%d_z=%s.pk'%(folder_out,grid,z)            
            
            
            # savefname = folder_out  + '/halo_density_data_dict_' + str(grid) + '.pk'
            # if os.path.exists(fout):  continue

            # compute the density field and save it to file
            if snapnum > 0:
                snapshot = '%s/%d/snapdir_%03d/snap_%03d'%(root,ji,snapnum,snapnum)
            else:
                snapshot = '%s/%d/ICs/ics' % (root, ji)
            df_cic = MASL.density_field_gadget(snapshot, ptypes, grid, MAS='CIC',
                                           do_RSD=False, axis=0, verbose=False)
            df_pylians_cic = df_cic/np.mean(df_cic, dtype=np.float64)-1.0

            # pos = readgadget.read_block(snapshot, "POS ", ptypes)/1e3 #positions in Mpc/h
            # df_uniform_cic_jax = cic_splitsuboxes(jnp.array(pos), BoxSize, grid, 3)
            # df_uniform_cic_jax = np.array(df_uniform_cic_jax/np.mean(df_uniform_cic_jax)-1.0)            
            
            
            df_ngp = MASL.density_field_gadget(snapshot, ptypes, grid, MAS='NGP',
                                           do_RSD=False, axis=0, verbose=False)
            df_pylians_ngp = df_ngp/np.mean(df_ngp, dtype=np.float64)-1.0            
            
                        

            for n_cnn in n_cnn_all:
                # find name of output file
                savefname_density_subvol = '%s/density_HR_subvol_m_res_%d_z=%s_nbatch_%d_nfilter_%d_ncnn_%d.pk'%(folder_out,grid,z,n_batch,n_filter,n_cnn)
                savefname_density_full = '%s/density_HR_full_m_res_%d_z=%s_nbatch_%d_nfilter_%d_ncnn_%d.pk'%(folder_out,grid,z,n_batch,n_filter,n_cnn)

                n_dim_red = (n_filter - 1) // 2
                n_pad = n_dim_red * n_cnn
                if n_cnn > 0:
                    df_cic_pad = np.pad(df_pylians_cic, n_pad, 'wrap')
                    # df_uniform_cic_pad = np.pad(df_uniform_cic_jax, n_pad, 'wrap')
                    df_ngp_pad = np.pad(df_pylians_ngp, n_pad, 'wrap')
                else:
                    df_cic_pad = df_pylians_cic
                    # df_uniform_cic_pad = df_uniform_cic_jax
                    df_ngp_pad = df_pylians_ngp

                # we want to split the df_pad into n_batch^3 sub-cubes, but centered on the original df simulation box
                xstart, ystart, zstart = n_pad, n_pad, n_pad
                subvol_size = grid // n_batch + 2 * n_pad
                nsubvol = n_batch**3
                save_subvol_density_cic_pad = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size))
                # save_subvol_density_uniform_cic_pad = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size))
                save_subvol_density_ngp_pad = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size))
                jc = 0
                from tqdm import tqdm
                for jx in (range(n_batch)):
                    for jy in range(n_batch):
                        for jz in range(n_batch):
                            # get the sub-cube
                            df_sub = df_cic_pad[xstart + jx * grid // n_batch - n_pad:xstart + (jx + 1) * grid // n_batch + n_pad,
                                            ystart + jy * grid // n_batch - n_pad:ystart + (jy + 1) * grid // n_batch + n_pad,
                                            zstart + jz * grid // n_batch - n_pad:zstart + (jz + 1) * grid // n_batch + n_pad]
                            # save the sub-cube
                            save_subvol_density_cic_pad[jc, ...] = df_sub

                            # df_sub = df_uniform_cic_pad[xstart + jx * grid // n_batch - n_pad:xstart + (jx + 1) * grid // n_batch + n_pad,
                            #                 ystart + jy * grid // n_batch - n_pad:ystart + (jy + 1) * grid // n_batch + n_pad,
                            #                 zstart + jz * grid // n_batch - n_pad:zstart + (jz + 1) * grid // n_batch + n_pad]
                            # # save the sub-cube
                            # save_subvol_density_uniform_cic_pad[jc, ...] = df_sub


                            df_sub = df_ngp_pad[xstart + jx * grid // n_batch - n_pad:xstart + (jx + 1) * grid // n_batch + n_pad,
                                            ystart + jy * grid // n_batch - n_pad:ystart + (jy + 1) * grid // n_batch + n_pad,
                                            zstart + jz * grid // n_batch - n_pad:zstart + (jz + 1) * grid // n_batch + n_pad]
                            # save the sub-cube
                            save_subvol_density_ngp_pad[jc, ...] = df_sub

                            jc += 1

                subvol_size = grid // n_batch
                nsubvol = n_batch**3
                save_subvol_density_cic_unpad = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size))
                # save_subvol_density_uniform_cic_unpad = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size))
                save_subvol_density_ngp_unpad = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size))
                jc = 0
                
                for jx in (range(n_batch)):
                    for jy in range(n_batch):
                        for jz in range(n_batch):
                            # get the sub-cube
                            save_subvol_density_cic_unpad[jc] = df_pylians_cic[jx * subvol_size:(jx + 1) * subvol_size,
                                                           jy * subvol_size:(jy + 1) * subvol_size,
                                                           jz * subvol_size:(jz + 1) * subvol_size]
                            
                            # save_subvol_density_uniform_cic_unpad[jc] = df_uniform_cic_jax[jx * subvol_size:(jx + 1) * subvol_size,
                            #                                   jy * subvol_size:(jy + 1) * subvol_size,
                            #                                     jz * subvol_size:(jz + 1) * subvol_size]
                            
                            
                            save_subvol_density_ngp_unpad[jc] = df_pylians_ngp[jx * subvol_size:(jx + 1) * subvol_size,
                                                           jy * subvol_size:(jy + 1) * subvol_size,
                                                           jz * subvol_size:(jz + 1) * subvol_size]                        
                            jc += 1


                saved_density_subvol = {
                    'density_cic_pad':save_subvol_density_cic_pad,
                    # 'density_uniform_cic_pad':save_subvol_density_uniform_cic_pad,
                    'density_ngp_pad':save_subvol_density_ngp_pad,
                    'density_cic_unpad':save_subvol_density_cic_unpad,
                    'density_ngp_unpad':save_subvol_density_ngp_unpad,
                    # 'density_uniform_cic_unpad':save_subvol_density_uniform_cic_unpad
                    }                        

                pk.dump(saved_density_subvol, open(savefname_density_subvol, 'wb'))

                saved_density_full = {
                    'density_cic_unpad_combined':df_pylians_cic,
                    # 'density_uniform_cic_unpad_combined':df_uniform_cic_jax,
                    'density_ngp_unpad_combined':df_pylians_ngp,                                
                    'density_cic_pad_combined':df_cic_pad,
                    'density_ngp_pad_combined':df_ngp_pad,
                    # 'density_uniform_cic_pad_combined':df_uniform_cic_pad                                            
                    }                        

                pk.dump(saved_density_full, open(savefname_density_full, 'wb'))



                
