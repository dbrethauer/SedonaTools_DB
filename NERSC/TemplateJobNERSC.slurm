#!/bin/bash
#SBATCH --job-name=sedona
#SBATCH --output=sedona_%j.out
#SBATCH --account=m2864
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --qos=regular
#SBATCH --time=10:00:00
#SBATCH --constraint=cpu

#***********************INSTRUCTIONS*******************************************************
#In order to couple each MPI process with OpenMP threads we have designed the following
#strategy:
#
# 1. First we fix one node in the --qos debug queue and fix a 2**n number of
#    --ntasks-per-node, starting with n=0.
#
# 2. Next, we compute the number --cpus-per-task=256/-ntask-per-node. This is
#    the number of virtual cores available to each MPI process on each node.
#    Each physical core is composed by two virtual cores in a (64+64) where each
#    NUMA domain will contain 64 physical cores.
#
# 3. Based on the available number of virtual cores, we obtain the compute of physical
#    cores and bind each OpenMP thread to each available physical core, using:
#    export OMP_NUM_THREADS. Also, a lower number may also be selected (in case of
#    memory shortage); however in principle we want to squeeze all the available resources
#    first.
#
# 4. Run the script and check the wall-clock-timestep. In perlmutter I use
#    grep Coarse <slurm_output>
#
# 5. Repeat the steps 1-4 until the perfect MPI/OpenMP balance is reached for the
#    choice of n.
#
# 6. Compare different amr.max_grid_size until the optimal value is reached. Usually is located
#    near half the Level 0 half_size. Furthermore, test several amr.blocking_factor sizes.
#
# 7. Finally, increase the number of nodes from 1, 2, 4, 8 and compare the
#    wall-clock time change. If the problem scales correctly, the wall-clock time will
#    go down by a factor of ~ 2, as we increase the nodes. However such scaling will break after
#    one particular bigger node choice. This is the perfect number of nodes we have to select.
#
#8. Run a chain of jobs using this script and ./chainslurm.sh

#export MPICH_MAX_THREAD_SAFETY=multiple

module purge
module load craype-x86-milan
module load PrgEnv-gnu
module load cray-hdf5-parallel 

export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

export SEDONA_EXEC=<INSERT_EXEC_PATH>
export INPUTS_FILE=<INSERT_INPUTS_PATH>

srun -n $((SLURM_NTASKS_PER_NODE * SLURM_NNODES)) -c ${SLURM_CPUS_PER_TASK} --cpu-bind=cores  ${SEDONA_EXEC} ${INPUTS_FILE}

