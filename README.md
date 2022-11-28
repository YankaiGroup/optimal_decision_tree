# [NeurIPS 2022] A Scalable Deterministic Global Optimization Algorithm for Training Optimal Decision Tree
This repository is the official implementation of [A Scalable Deterministic Global Optimization Algorithm for Training Optimal Decision Tree](https://nips.cc/Conferences/2022/Schedule?showEvent=53587). 

## Requirements
* Julia v1.7.2
* CPLEX v20.1.0
* openmpi v4.1.1 
    * We also tested Microsoft MPI v10.1.12498.18. Please test other MPI implementations by yourself. 
* Julia Packages
    * CPLEX v0.9.3
    * CSV v0.10.4
    * CategoricalArrays v0.10.6
    * Clustering v0.14.2
    * DataFrames v1.3.4
    * DecisionTree v0.10.12
    * Distances v0.10.7
    * Distributions v0.25.65
    * JuMP v1.1.1
    * Luxor v3.4.0
    * MLDataUtils v0.5.4
    * MPI v0.19.2
    * Plots v1.31.2
    * RDatasets v0.7.6
    * StaticArrays v1.5.1
    * StatsBase v0.33.18
    * LinearAlgebra
    * Printf
    * Random
    * SparseArrays
    * Statistics

## File list
###  Source Files - src/
* bb_func.jl - branch and bound functions.

* lb_func.jl - lower bound calculation functions.

* ub_func.jl - upper bound calculation functions.

* opt_func.jl - local and global optimization functions.

* data_process.jl - data preprocess and result comparison.

* branch.jl - branching functions.

* Nodes.jl - BB node struct.

* Trees.jl - Tree variable struct for calculation.

* groups.jl - functions for decomosing dataset into groups (subproblems).

* parallel.jl - functions for wrapped mpi communication and calculation.
### Test Files - test/
* test.jl - testing codes for datasets with base-dim, base-glb, and BB+LD methods.

* real-l.sh - example shell file to run the code. Line 24 indicate the parallel command and line 25 indicate the serial command. The parallel version is executed through MPI. 
### Dataset Files - data/
All small and medium dataset are listed in the folder. The link of large datasets is listed in file large-datasets/readme. 
## Evaluation
Please refer to either line 28-33 of real-l.jl or line 25-33 of test.jl file for the meaning of each input parameter.

