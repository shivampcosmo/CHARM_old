# %pip install Pylians
import numpy as np
import sys,os
import readgadget
import MAS_library as MASL
import pickle as pk
import readfof
import matplotlib
from tqdm import tqdm
from nbodykit.source.catalog.file import BigFileCatalog
import nbodykit.lab as nb

root_in = '/mnt/home/fvillaescusa/ceph/Quijote/Snapshots/latin_hypercube_HR/'
ptypes       = [1]
root_out = '/mnt/home/spandey/ceph/Quijote/data_NGP_self_LH/'
grids         = [128]
BoxSize = 1000.0 #Mpc/h ; size of box
n_batch = 8
n_filter = 3
n_cnn_all = [0,4]
n_sim_array = np.arange(0,1100)
# snap_num_array = [4, 3, -1]
snap_num_array = [4, 3, -1]
BoxSize = 1000.0

def save_cic_densities(ji):
# for ji in tqdm(n_sim_array):
    # print('doing sim: ' + str(ji))
    # print('doing sim: ' + str(ji))
    for grid in grids:
        # print('doing res: ' + str(grid))
        for snapnum in snap_num_array:
            z = {4:0, 3:0.5, 2:1, 1:2, 0:3, -1: 127}[snapnum]

            # create output folder if it does not exists
            folder_out = '%s/%d'%(root_out,ji)
            if not(os.path.exists(folder_out)):
                os.system('mkdir %s'%folder_out)

            
            # savefname_halos_subvol = '%s/halos_HR_subvol_res_%d_z=%s.pk'%(folder_out,grid,z)
            # savefname_halos_full = '%s/halos_HR_full_res_%d_z=%s.pk'%(folder_out,grid,z)            
            
            
            # savefname = folder_out  + '/halo_density_data_dict_' + str(grid) + '.pk'
            # if os.path.exists(fout):  continue

            # compute the density field and save it to file
            if snapnum > 0:
                snapshot = '%s/%d/snapdir_%03d/snap_%03d'%(root_in,ji,snapnum,snapnum)
            else:
                snapshot = '%s/%d/ICs/ics' % (root_in, ji)
            df_cic = MASL.density_field_gadget(snapshot, ptypes, grid, MAS='CIC',
                                           do_RSD=False, axis=0, verbose=False)
            df_pylians_cic = df_cic/np.mean(df_cic, dtype=np.float64)-1.0

            # pos = readgadget.read_block(snapshot, "POS ", ptypes)/1e3 #positions in Mpc/h
            # df_uniform_cic_jax = cic_splitsuboxes(jnp.array(pos), BoxSize, grid, 3)
            # df_uniform_cic_jax = np.array(df_uniform_cic_jax/np.mean(df_uniform_cic_jax)-1.0)            
            
            
            # df_ngp = MASL.density_field_gadget(snapshot, ptypes, grid, MAS='NGP',
            #                                do_RSD=False, axis=0, verbose=False)
            # df_pylians_ngp = df_ngp/np.mean(df_ngp, dtype=np.float64)-1.0            
            file_exists = True
            for n_cnn in n_cnn_all:
                # find name of output file
                savefname_density_subvol = '%s/density_HR_subvol_m_res_%d_z=%s_nbatch_%d_nfilter_%d_ncnn_%d.pk'%(folder_out,grid,z,n_batch,n_filter,n_cnn)
                savefname_density_full = '%s/density_HR_full_m_res_%d_z=%s_nbatch_%d_nfilter_%d_ncnn_%d.pk'%(folder_out,grid,z,n_batch,n_filter,n_cnn)
                # file_names_all.append(savefname_density_subvol)
                # file_names_all.append(savefname_density_full)
                if not(os.path.exists(savefname_density_subvol)):
                    file_exists = False
                    break

            if not file_exists:
                for n_cnn in n_cnn_all:
                    # find name of output file
                    savefname_density_subvol = '%s/density_HR_subvol_m_res_%d_z=%s_nbatch_%d_nfilter_%d_ncnn_%d.pk'%(folder_out,grid,z,n_batch,n_filter,n_cnn)
                    savefname_density_full = '%s/density_HR_full_m_res_%d_z=%s_nbatch_%d_nfilter_%d_ncnn_%d.pk'%(folder_out,grid,z,n_batch,n_filter,n_cnn)

                    n_dim_red = (n_filter - 1) // 2
                    n_pad = n_dim_red * n_cnn
                    if n_cnn > 0:
                        df_cic_pad = np.pad(df_pylians_cic, n_pad, 'wrap')
                        # df_uniform_cic_pad = np.pad(df_uniform_cic_jax, n_pad, 'wrap')
                        # df_ngp_pad = np.pad(df_pylians_ngp, n_pad, 'wrap')
                    else:
                        df_cic_pad = df_pylians_cic
                        # df_uniform_cic_pad = df_uniform_cic_jax
                        # df_ngp_pad = df_pylians_ngp

                    # we want to split the df_pad into n_batch^3 sub-cubes, but centered on the original df simulation box
                    xstart, ystart, zstart = n_pad, n_pad, n_pad
                    subvol_size = grid // n_batch + 2 * n_pad
                    nsubvol = n_batch**3
                    save_subvol_density_cic_pad = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size))
                    # save_subvol_density_uniform_cic_pad = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size))
                    # save_subvol_density_ngp_pad = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size))
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


                                # df_sub = df_ngp_pad[xstart + jx * grid // n_batch - n_pad:xstart + (jx + 1) * grid // n_batch + n_pad,
                                #                 ystart + jy * grid // n_batch - n_pad:ystart + (jy + 1) * grid // n_batch + n_pad,
                                #                 zstart + jz * grid // n_batch - n_pad:zstart + (jz + 1) * grid // n_batch + n_pad]
                                # # save the sub-cube
                                # save_subvol_density_ngp_pad[jc, ...] = df_sub

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
                                
                                
                                # save_subvol_density_ngp_unpad[jc] = df_pylians_ngp[jx * subvol_size:(jx + 1) * subvol_size,
                                #                                jy * subvol_size:(jy + 1) * subvol_size,
                                #                                jz * subvol_size:(jz + 1) * subvol_size]                        
                                jc += 1


                    saved_density_subvol = {
                        'density_cic_pad':save_subvol_density_cic_pad,
                        # 'density_uniform_cic_pad':save_subvol_density_uniform_cic_pad,
                        # 'density_ngp_pad':save_subvol_density_ngp_pad,
                        'density_cic_unpad':save_subvol_density_cic_unpad,
                        # 'density_ngp_unpad':save_subvol_density_ngp_unpad,
                        # 'density_uniform_cic_unpad':save_subvol_density_uniform_cic_unpad
                        }                        

                    pk.dump(saved_density_subvol, open(savefname_density_subvol, 'wb'))

                    saved_density_full = {
                        'density_cic_unpad_combined':df_pylians_cic,
                        # 'density_uniform_cic_unpad_combined':df_uniform_cic_jax,
                        # 'density_ngp_unpad_combined':df_pylians_ngp,                                
                        'density_cic_pad_combined':df_cic_pad,
                        # 'density_ngp_pad_combined':df_ngp_pad,
                        # 'density_uniform_cic_pad_combined':df_uniform_cic_pad                                            
                        }                        

                    pk.dump(saved_density_full, open(savefname_density_full, 'wb'))



                

def save_Cls_batch(jrank, njobs):
    ni = 0
    nf = 1100
    lhs_all = np.arange(ni, nf)
    lhs_all_split = np.array_split(lhs_all, njobs)
    lhs_jrank = lhs_all_split[jrank]

    # for lhs in tqdm(lhs_jrank):
    for lhs in (lhs_jrank):
        save_cic_densities(lhs)
        # get_cutout(lhs, snv, jsnv, 1, 0)
        # get_cutout(lhs, snv, jsnv, 2, 0)
        # get_cutout(lhs, snv, jsnv, 3, 0)


from mpi4py import MPI
if __name__ == '__main__':
    run_count = 0
    n_jobs = 128

    while run_count < n_jobs:
        comm = MPI.COMM_WORLD
        print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
        if (run_count + comm.rank) < n_jobs:
            save_Cls_batch(comm.rank, n_jobs)
        run_count += comm.size
        comm.bcast(run_count, root=0)
        comm.Barrier()


# # salloc -N 4 -C haswell -q interactive -t 04:00:00 -L SCRATCH
# # srun --nodes=4 --tasks-per-node=32 --cpu-bind=cores python process_density_NGP_quijote.py
