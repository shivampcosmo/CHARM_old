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

import cython

cimport numpy as np
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
        


root_in = '/mnt/home/fvillaescusa/ceph/Quijote/Snapshots/latin_hypercube_HR/'
ptypes       = [1]
# mass_types = ['rockstar_200c']
mass_type = 'rockstar_200c'
root_out = '/mnt/home/spandey/ceph/Quijote/data_NGP_self_LH/'
grids         = [128]
BoxSize = 1000.0 #Mpc/h ; size of box
n_batch = 8
n_filter = 3
n_cnn_all = [0,4]
n_sim_array = np.arange(0,1100)
# snap_num_array = [4, 3, -1]
snap_num_array = [3]
BoxSize = 1000.0
Mmin_cut = 1e13
Mmin_cut_str = '1e13'

def save_cic_densities(ji):
# for ji in tqdm(n_sim_array):
    for grid in grids:
        # print('doing res: ' + str(grid))
        for snapnum in snap_num_array:
            z = {4:0, 3:0.5, 2:1, 1:2, 0:3, -1: 127}[snapnum]

            # create output folder if it does not exists
            folder_out = '%s/%d'%(root_out,ji)
            if not(os.path.exists(folder_out)):
                os.system('mkdir %s'%folder_out)

            
            savefname_halos_subvol = '%s/halos_HR_%s_lgMmincut_%s_subvol_res_%d_z=%s.pk'%(folder_out,mass_type,Mmin_cut_str,grid,z)
            savefname_halos_full = '%s/halos_HR_%s_lgMmincut_%s_full_res_%d_z=%s.pk'%(folder_out,mass_type,Mmin_cut_str,grid,z)            
            
            if snapnum > 0:
                if 'rockstar' in mass_type:
                    snap_dir_base = '/mnt/home/fvillaescusa/ceph/Quijote/Halos/Rockstar/latin_hypercube_HR'
                    snapdir = snap_dir_base + '/' + str(ji)  #folder hosting the catalogue
                    rockstar = np.loadtxt(snapdir + '/out_' + str(snapnum) + '_pid.list')
                    with open(snapdir + '/out_' + str(snapnum) + '_pid.list', 'r') as f:
                        lines = f.readlines()
                    header = lines[0].split()
                    # get the properties of the halos
                    pos_h_truth = rockstar[:,header.index('X'):header.index('Z')+1]
                    if mass_type == 'rockstar_vir':
                        index_M = header.index('Mvir')                    
                        mass_truth = rockstar[:,index_M]  #Halo masses in Msun/h
                    if mass_type == 'rockstar_200c':
                        index_M = header.index('M200c')                    
                        mass_truth = rockstar[:,index_M]  #Halo masses in Msun/h
                if 'fof' in mass_type:
                    # snap_dir_base = '/mnt/home/spandey/ceph/Quijote/fiducial_HR_new/Halos/FoF'
                    snapdir = snap_dir_base + '/' + str(ji)  #folder hosting the catalogue
                    FoF = readfof.FoF_catalog(snapdir, snapnum, long_ids=False, swap=False, SFR=False, read_IDs=False)
                    # get the properties of the halos
                    pos_h_truth = FoF.GroupPos / 1e3  #Halo positions in Mpc/h
                    mass_truth = FoF.GroupMass * 1e10  #Halo masses in Msun/h

    
                lgMass_truth = np.log10(mass_truth)
                indsel = np.where(mass_truth > Mmin_cut)[0]
                print(grid, len(indsel), len(mass_truth), np.amin(lgMass_truth), np.log10(Mmin_cut))
                pos_h_truth = pos_h_truth[indsel]
                lgMass_truth = lgMass_truth[indsel]

                Nhalos = np.float32(np.zeros((grid, grid, grid)))
                MASL.NGP(np.float32(pos_h_truth), Nhalos, BoxSize)
                print('mass type: ', str(mass_type), ', max number of halos:', np.amax(Nhalos))

                if grid == 64:
                    nMax_h = 30  # maximum number of halos expected in a cell
                elif grid == 128:
                    nMax_h = 10
                elif grid == 256:
                    nMax_h = 8
                elif grid == 512:
                    nMax_h = 3
                elif grid == 1024:
                    nMax_h = 2
                else:
                    print('nside not supported')
                    sys.exit()

                dfhalo_ngp_wmass = np.float32(np.zeros((grid, grid, grid, nMax_h)))
                NGP_mass(np.float32(pos_h_truth), np.float32(lgMass_truth), dfhalo_ngp_wmass, BoxSize)


                M_halos = np.flip(np.sort(dfhalo_ngp_wmass, axis=-1), axis=-1)


                # now split it into nbatches each side

                subvol_size = grid // n_batch
                nsubvol = n_batch**3
                save_subvol_Nhalo = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size))
                save_subvol_Mhalo = np.zeros((nsubvol, subvol_size, subvol_size, subvol_size, nMax_h))

                jc = 0
                from tqdm import tqdm
                for jx in (range(n_batch)):
                    for jy in range(n_batch):
                        for jz in range(n_batch):
                            # get the sub-cube
                            save_subvol_Nhalo[jc] = Nhalos[jx * subvol_size:(jx + 1) * subvol_size,
                                                        jy * subvol_size:(jy + 1) * subvol_size,
                                                        jz * subvol_size:(jz + 1) * subvol_size]
                            save_subvol_Mhalo[jc] = M_halos[jx * subvol_size:(jx + 1) * subvol_size,
                                                            jy * subvol_size:(jy + 1) * subvol_size,
                                                            jz * subvol_size:(jz + 1) * subvol_size]
                            # save_subvol_density_cic_unpad[jc] = df_pylians_cic[jx * subvol_size:(jx + 1) * subvol_size,
                            #                                jy * subvol_size:(jy + 1) * subvol_size,
                            #                                jz * subvol_size:(jz + 1) * subvol_size]
                            # save_subvol_density_ngp_unpad[jc] = df_pylians_ngp[jx * subvol_size:(jx + 1) * subvol_size,
                            #                                jy * subvol_size:(jy + 1) * subvol_size,
                            #                                jz * subvol_size:(jz + 1) * subvol_size]                        
                            jc += 1

                saved_halos_subvol = {
                    'N_halos': save_subvol_Nhalo,
                    'M_halos': save_subvol_Mhalo,
                    }    
                pk.dump(saved_halos_subvol, open(savefname_halos_subvol, 'wb'))

                saved_halos_full = {
                    'N_halos_combined': Nhalos,
                    'M_halos_combined': M_halos,
                    }    
                pk.dump(saved_halos_full, open(savefname_halos_full, 'wb'))


                

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
# # srun --nodes=4 --tasks-per-node=32 --cpu-bind=cores python process_density_halo_NGP_fastpm.py
