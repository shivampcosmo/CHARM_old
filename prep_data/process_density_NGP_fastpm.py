# %pip install Pylians
import numpy as np
import sys,os
# import readgadget
import MAS_library as MASL
import pickle as pk
# import readfof
# import matplotlib
from tqdm import tqdm
from nbodykit.source.catalog.file import BigFileCatalog
import nbodykit.lab as nb



def save_cic_densities(ji):
    try:
        root_in = '/mnt/ceph/users/cmodi/fastpm-shivam/LH_HR/'
        root_out = '/mnt/home/spandey/ceph/Quijote/data_NGP_self_fastpm_LH/'
        grids         = [128]
        BoxSize = 1000.0 #Mpc/h ; size of box
        n_batch = 8
        n_filter = 3
        n_cnn_all = [0,4]
        # snap_num_array = [4, 3, -1]
        snap_num_array = [3]
        BoxSize = 1000.0    
        for grid in grids:
            # print('doing res: ' + str(grid))
            for snapnum in snap_num_array:
                z = {4:0, 3:0.5, 2:1, 1:2, 0:3, -1: 99}[snapnum]

                # create output folder if it does not exists
                folder_out = '%s/%d'%(root_out,ji)
                if not(os.path.exists(folder_out)):
                    os.system('mkdir %s'%folder_out)

                # file_names_all = []
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
                
                    
                # if not file_exists:
                if 1:                
                    if snapnum == 4:
                        fname = '%s/%d' % (root_in, ji) + '/fastpm_B2_1.0000'
                        # df = nb.BigFileCatalog('/mnt/home/spandey/ceph/fastpm/LH_HR/' + str(js) + '/fastpm_B2_' + a, dataset='1')        
                        # pos = np.array(df['Position'], dtype=np.float64)
                    elif snapnum == 3:
                        fname = '%s/%d' % (root_in, ji) + '/fastpm_B2_0.6667'
                    elif snapnum == -1:
                        fname = '%s/%d' % (root_in, ji) + '/fastpm_B2_0.0100'                
                        # snapshot = '%s/%d' % (root_z99, ji)
                        # df_cic = np.load('%s/rho.npy'%(snapshot))
                    # df_pylians_cic = df_cic

                    # pos = np.array(df['Position'], dtype=np.float64)
                    # df.columns
                    df = nb.BigFileCatalog(fname, dataset='1')        
                    pos = np.array(df['Position'], dtype=np.float64)                
                    
                    # pos = np.array(df['Position'], dtype=np.float64)
                    df_pylians = np.zeros((grid,grid,grid), dtype=np.float32)
                    MASL.MA(np.float32(pos), df_pylians, BoxSize, 'CIC', verbose=False)
                    df_pylians_cic = df_pylians/np.mean(df_pylians, dtype=np.float64)-1.0


            
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
                                    
                                    jc += 1


                        saved_density_subvol = {
                            'density_cic_pad':save_subvol_density_cic_pad
                            }                        

                        pk.dump(saved_density_subvol, open(savefname_density_subvol, 'wb'))

                        saved_density_full = {
                            'density_cic_unpad_combined':df_pylians_cic
                            }                        

                        pk.dump(saved_density_full, open(savefname_density_full, 'wb'))


    except:
        pass

import multiprocessing as mp
if __name__ == '__main__':
    # n_sims = 1100
    n_sims_offset = 1100
    n_sims = 900
    n_cores = mp.cpu_count()
    print(n_cores)

    # Create a pool of worker processes
    pool = mp.Pool(processes=n_cores)

    # Distribute the simulations across the available cores
    sims_per_core = n_sims // n_cores
    sim_ranges = [(n_sims_offset + i * sims_per_core, n_sims_offset + (i + 1) * sims_per_core) for i in range(n_cores)]

    # Handle any remaining simulations
    remaining_sims = n_sims % n_cores
    if remaining_sims > 0:
        sim_ranges[-1] = (sim_ranges[-1][0], sim_ranges[-1][1] + remaining_sims)

    # Run save_cic_densities function for each simulation range in parallel
    results = [pool.apply_async(save_cic_densities, args=(ji,)) for sim_range in sim_ranges for ji in range(*sim_range)]

    # Wait for all tasks to complete
    [result.get() for result in results]

    # Close the pool and wait for tasks to finish
    pool.close()
    pool.join()

# def save_Cls_batch(jrank, njobs):
#     ni = 0
#     nf = 1100
#     lhs_all = np.arange(ni, nf)
#     lhs_all_split = np.array_split(lhs_all, njobs)
#     lhs_jrank = lhs_all_split[jrank]

#     # for lhs in tqdm(lhs_jrank):
#     for lhs in (lhs_jrank):
#         save_cic_densities(lhs)
#         # get_cutout(lhs, snv, jsnv, 1, 0)
#         # get_cutout(lhs, snv, jsnv, 2, 0)
#         # get_cutout(lhs, snv, jsnv, 3, 0)


# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

# n_sims = 1100
# sims_per_rank = n_sims // size
# sim_starts = [rank * sims_per_rank + min(rank, n_sims % size) for rank in range(size)]
# sim_ends = [sim_starts[rank] + sims_per_rank + (rank < n_sims % size) for rank in range(size)]

# for ji in range(sim_starts[rank], sim_ends[rank]):
#    save_cic_densities(ji)
# if __name__ == '__main__':
#     run_count = 0
#     n_jobs = 20

#     while run_count < n_jobs:
#         comm = MPI.COMM_WORLD
#         print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
#         if (run_count + comm.rank) < n_jobs:
#             save_Cls_batch(comm.rank, n_jobs)
#         run_count += comm.size
#         comm.bcast(run_count, root=0)
#         comm.Barrier()


# # # salloc -N 4 -t 04:00:00
# # # srun --nodes=4 --tasks-per-node=20 python process_density_halo_NGP_fastpm.py
