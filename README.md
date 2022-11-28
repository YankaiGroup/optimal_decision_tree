# [NeurIPS 2022] Global Optimal K-Medoids Clustering of One Million Samples
This repository is the official implementation of [Global Optimal K-Medoids Clustering of One Million Samples](https://nips.cc/Conferences/2022/Schedule?showEvent=53593). 

## Requirements
* Julia v1.6.1
* CPLEX v20.1.0
* openmpi v4.1.1 
    * We also tested Microsoft MPI v10.1.12498.18. Please test other MPI implementations by yourself. 
* Julia Packages
    * CPLEX v0.7.8
    * CSV v0.9.11
    * Clustering v0.14.2
    * DataFrames v1.3.1
    * Distances v0.10.7
    * Distributions v0.25.37
    * JuMP v0.21.10
    * MPI v0.19.2
    * MathOptInterface v0.9.22
    * Metaheuristics v3.2.3
    * RDatasets v0.7.6
    * TimerOutputs v0.5.13
    * LinearAlgebra
    * Printf
    * Random
    * Statistics

## File list
###  Source Files - src_mpi/
* bb_functions.jl - branch and bound functions.

* lb_functions.jl - lower bound calculation functions.

* ub_functions.jl - upper bound calculation functions.

* opt_functions.jl - local and global optimization functions.

* data_process.jl - data preprocess and result evaluation and plotting functions.

* branch.jl - branching functions.

* Nodes.jl - Node struct of BB procedure.

* probing.jl - functions for Probing bound tightening.

* fbbt.jl - functions for Feasibility Based Bound Tightening.
### Test Files - src_mpi/test

* testing_real_mpi.jl - testing codes for datasets with LD, BB+Basic, and BB+LD methods.

* testing_real_cpx.jl - testing codes for real-world data with CPLEX.

## Evaluation
Please refer to line 9-16 of testing_*.jl file for the meaning of each input parameter 
* Dataset List
    * "iris" "seeds" "glass" "BM_FL" "UK_L" "HF_L" "Who_FL" "HCV_L" "Abs_FL" "TR_FL" "SGC_FL" "hemi" "pr2392" "TRR_FL" "AC_FL" "rds_cnt" "HTRU2_L" "GT_FL" "rds" "KEGG_FL" "urbanGB_L_10" "rng_agr" "urbanGB_L" "spnet3D" "retail" "retail_II" "syn_1E7_2_3" "USC1990"
* Examples for the serial mode
```shell
julia src_mpi/test/test_real_mpi.jl 3 hemi BB+LD true fbbt_analytic 3 true true

julia src_mpi/test/test_real_mpi.jl 3 hemi LD true fbbt_analytic 3 true true

julia src_mpi/test/test_real_mpi.jl 3 hemi BB+Basic true fbbt_analytic 3 false false

julia src_mpi/test/test_real_cpx.jl 3 hemi
```

* Examples for the parallel mode
```shell
mpiexec -n 4 julia src_mpi/test/test_real_mpi.jl 3 hemi BB+LD true fbbt_analytic 3 true true
```
## Results
Here is the results of the hemi dataset with 1955 samples, 7 dimensions, and cluster number K=3. We executed all the experiments on a high-performance computing cluster, of which each node contains 40 Intel cores at 2.4 GHz and 202 GB RAM. Besides, the Heuristic results were recorded in the corresponding BB+LD logs.
|   Method  |    UB    |  Nodes | Gap      (%) | Time      (s) |
|:---------:|:--------:|:------:|:------------:|:-------------:|
| Heuristic | 9.91E+06 |    -   |       -      |       -       |
|   CPLEX   | 9.91E+06 |    1   |     ≤0.10    |      2044     |
|     LD    | 9.91E+06 |    1   |     12.45    |      3176     |
|  BB+Basic | 9.91E+06 | 1.9E+6 |     4.32     |       4h      |
|   BB+LD   | 9.91E+06 |   63   |     ≤0.10    |       97      |
