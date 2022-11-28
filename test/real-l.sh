#! /bin/bash
#SBATCH --nodes=25
#SBATCH --ntasks-per-node=40
#SBATCH -t 0-8:30
#SBATCH --array=0,1
#SBATCH --output=info-%x-%a-2-CF+MILP+SG.out

cd ${SLURM_SUBMIT_DIR}

module load NiaEnv/2019b
#module load gcc/9.2.0 openmpi/4.0.3
module load intel/2019u4 intelmpi/2019u4
#module load ddt # load this module to prevent error of julia module loading error
module load julia/1.7.2
module load mycplex/20.1.0
module use /scinet/niagara/software/commercial/modules
module load gurobi/9.0.2

# seeds number for multi-run
seeds=("1" "2" "3" "4" "5")
# 0-1 # large datasets
datasets=("Skin_NonSkin.txt" "HTS_processed") # 

mpiexec -n ${SLURM_NTASKS} julia test/test.jl 2 CF+MILP+SG ${seeds[0]} par ${datasets[${SLURM_ARRAY_TASK_ID}]} > ${datasets[${SLURM_ARRAY_TASK_ID}]}-sd${seeds[0]}-2-CMS-${SLURM_NTASKS}.out
#julia test/test.jl 2 CF+MILP+SG ${seeds[0]} sl ${datasets[${SLURM_ARRAY_TASK_ID}]} > ${datasets[${SLURM_ARRAY_TASK_ID}]}-sd${seeds[0]}-2-CMS-${SLURM_NTASKS}.out

:<<EOF
    # input argument of julia
    # args[1] maximum depth of the tree
    # args[2] lower bound method: base-dim, base-glb, CF+MILP+SG
    # args[3] random seed value
    # args[4] mode (parallel or serial)
    # args[5] dataset name
EOF
