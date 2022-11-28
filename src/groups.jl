module groups

using Random, Distributions
using LinearAlgebra, SparseArrays, Statistics, StatsBase
using MLDataUtils, Clustering
using JuMP

using Distributed, SharedArrays
using parallel
@everywhere using Trees, bound, Nodes

export n_groups

tol = 1e-6
max_iter = 1000


############## auxilary functions for adaptive sub-grouping ##############
function unique_inverse(A::AbstractArray)
    out = Array{eltype(A)}(undef, 0)
    out_idx = Array{Vector{Int}}(undef, 0)
    seen = Dict{eltype(A), Int}()
    for (idx, x) in enumerate(A)
        if !in(x, keys(seen))
            seen[x] = length(seen) + 1
            push!(out, x)
            push!(out_idx, Int[])
        end
        push!(out_idx[seen[x]], idx)
    end
    out, out_idx
end

# this is the function for grouping a cluster, and can avoid failure when a sub-cluster has size < ngroups
function strGrp_nofill(assign, ngroups)
    l, c = unique_inverse(assign) # get label and cluster index
    groups = [[] for i=1:ngroups]
    ng = 0
    for i in 1:length(l)
        p = c[i][randperm(length(c[i]))] # first shuffling the index to introduce the randomness
        for j in eachindex(p) #1:length(p)	
            if ng == ngroups
                ng = 0;
            end	  
            ng += 1;
            push!(groups[ng], p[j]);
        end
    end
    return groups
end

# grouping function that stratified on assign and select data evenly by applying kmeans clustering on each cluster
# the sampling will have a bug in kmeans grouping with sample function when the smallest cluster only has p data points, which p < 2*ngroups
function kmeans_group(X, c, z, D, ngroups)
    Random.seed!(123)
    clst_label, clst_idx = Trees.getleafassign(c,z,D) # clst_label is [1,2,3], clst_idx is [[1,2],[3,4]]
    groups = [Int64[] for i=1:ngroups]
    # This may happen if at particular node, some centers are too ridiculous that no sample are close to them.
    for i in eachindex(clst_idx) #1:length(clst_idx) 
        #println(clst_idx[i])
        if length(clst_idx[i]) <= ngroups # if this cluster only have one sample, put it directly to group 1
            for j in 1:length(clst_idx[i])
                append!(groups[j],clst_idx[i][j]) 
            end
        else
            # using floor to guarantee one cluster can assign points to every group
            k_sub = floor(Int, length(clst_idx[i])/ngroups) # get number of sub-cluster for each cluster, the higher the k_sub, the more sparse of the subgroup
            if k_sub <=1
                k_sub = 2 # here may have the problem ksub in one cluster separate into k sub groups
            end
            # try 5 trials of kmeans to get the best assignment
            #Random.seed!(1)
            n_trial = 5
            mini_cost = Inf
            X_sub = X[:,clst_idx[i]]
            assign = nothing
            for tr = 1:n_trial
                clst_rlt = kmeans(X_sub, k_sub) # get sub-cluster from one cluster
                if clst_rlt.totalcost <= mini_cost
                    mini_cost = clst_rlt.totalcost
                    assign = clst_rlt.assignments
                end
            end
            clst_groups = strGrp_nofill(assign, ngroups) # get the grouping label from one cluster
            
            for j in eachindex(clst_groups) #1:length(clst_groups) # we put index of points in sub-clusters from cluster i that belongs to group j
                append!(groups[j], clst_idx[i][clst_groups[j]]);
            end
        end
    end
    return groups
end

function grouping(X, tree, D, ngroups)
    nz,Tl = size(tree.z)
    p,n = size(X)
    if nz != n
        z = bound.warm_start_z(X, tree.a, tree.b, D)
    else
        z = tree.z
    end
    return kmeans_group(X, tree.c, z, D, ngroups)
end


# determine the number of groups
function n_groups(X, p::Int, n::Int, k::Int, warm_start::Tree, D::Int, method::Vector{SubString{String}})
    # grouping calculation
    if "SG" in method # SG
        # if D = 2, spl per gp <= 100
        if n < 50
            ngroups = 2
        else
            
            if D >2
                if n <= 5000 && n >= 1000
                    gp_size=100
                else
                    gp_size = 50
                end
            else
                #gp_size = 50
                
                cnst = 150 # 319 for small and medium data and 150 for large data
                gp_size = 1/2^D*(cnst+p+2)-(p+2+k)
                if gp_size <= 0
                    gp_size = max(30,k)
                end
                
            end
            
            ngroups = Int(round(n/gp_size)) # Int(round(n/200*2^D)) #
        end
        parallel.root_println("ngroups: $ngroups")
        groups = grouping(X, warm_start, D, ngroups)
        #println(length.(groups))
    else # if the method is LD or closed-form, then grouping is based on single samples
        ngroups = n
        groups = [[i] for i=1:ngroups]
    end
    return ngroups, groups
end

function group_generation(X::Union{Nothing, Matrix{Float64}}, Y::Union{Nothing, Matrix{Float64}}, p::Int64, n_all::Int64, K::Int64, D::Int64, warm_start::Tree, method::Vector{SubString{String}})
    if parallel.is_root()
        #if "SG" in mtd
        ngroups_all, groups_all = n_groups(X, p, n_all, K, warm_start, D, method)
        #println("groups_all: $groups_all")
        # generate group based X
        X_gp = Matrix{Float64}[]
        Y_gp = Matrix{Float64}[]
        for i in eachindex(groups_all) #1:length(groups_all)
            push!(X_gp, X[:,groups_all[i]])
            push!(Y_gp, Y[:,groups_all[i]])
        end
    else
        ngroups_all = nothing
        groups_all = nothing
        X_gp = nothing
        Y_gp = nothing
    end
    ngroups_all = parallel.bcast(ngroups_all)
    #parallel.create_comm(ngroups_all)
    groups_all = parallel.bcast(groups_all)
    return X_gp, Y_gp, ngroups_all, groups_all
end

function group_generation_rand(X::Union{Nothing, Matrix{Float64}}, Y::Union{Nothing, Matrix{Float64}}, n_all::Int64, iter::Int64=0)
    if parallel.is_root()
        X_rand = Matrix{Float64}[]
        Y_rand = Matrix{Float64}[]
        ng_all_rand = parallel.nprocs() <= 3 ? 6 : parallel.nprocs() # if nprocs <= 3, then totally 6 groups are generated.
        gp_all_rand = Vector{Int64}[]
        for i in 1:ng_all_rand::Int
            Random.seed!(i*(iter+1)) # here the group sample are selected only for data in each core
            group = sample(1:n_all, 50, replace = true) # true for bootstraping, false for exclusive selection
            group = unique(group)
            push!(gp_all_rand, group)
            push!(X_rand, X[:,group])
            push!(Y_rand, Y[:,group])
        end
    else
        X_rand = nothing
        Y_rand = nothing
        ng_all_rand = nothing
        gp_all_rand = nothing
    end
    ng_all_rand = parallel.bcast(ng_all_rand)
    gp_all_rand = parallel.bcast(gp_all_rand)
    return X_rand, Y_rand, ng_all_rand, gp_all_rand
end

function group_distribute(X_gp::Union{Nothing, Vector{Matrix{Float64}}}, Y_gp::Union{Nothing, Vector{Matrix{Float64}}}, ngroups_all::Int64)
    # get partitionlist
    parallel.partition_concat(ngroups_all)
    gp_list = parallel.getpartition()
    # spread X to each process according to gp_list
    X_gp = parallel.spread(X_gp)
    Y_gp = parallel.spread(Y_gp)
    # if X_gp is nothing, then this process is not used in parallel computing, so we set X_proc to nothing.
    if length(X_gp) == 0
        X_proc = Matrix{Float64}[]
        Y_proc = Matrix{Float64}[]
    else
        X_proc = view(reduce(hcat, X_gp), :,:)
        Y_proc = view(reduce(hcat, Y_gp), :,:)
    end
    
    return X_gp, Y_gp, X_proc, Y_proc, gp_list
end

function groups_on_proc(groups_all::Vector{Vector{Int64}}, gp_list::Vector{Int64})
    gp_length = map(x->length(x), groups_all[gp_list]) # group index for each process
    gp_accu = accumulate(+, gp_length)
    pushfirst!(gp_accu, 0)
    groups = UnitRange{Int64}[]
    for i in 1:(length(gp_accu)-1)::Int64
        push!(groups, (gp_accu[i]+1):gp_accu[i+1])
    end
    return groups
end

function group_trees_init(warm_start::Tree, ngroups::Int64)
    group_trees = Tree[]
    for i in 1:ngroups::Int64
        t = Tree(warm_start.a, warm_start.b, warm_start.c, warm_start.d, nothing, warm_start.D)
        push!(group_trees, t)
    end
    return group_trees
end

function proc_data_preparation(X::Union{Nothing, Matrix{Float64}}, Y::Union{Nothing, Matrix{Float64}}, p::Int64, n_all::Int64, K::Int64, D::Int64, warm_start::Tree, method::Vector{SubString{String}}, rand::Bool=false, iter::Int64=0, val::Int64=0)
    # get group data
    if !rand
        X_gp, Y_gp, ngroups_all, groups_all = group_generation(X, Y, p, n_all, K, D, warm_start, method)
    else
        X_gp, Y_gp, ngroups_all, groups_all = group_generation_rand(X, Y, n_all, iter)
    end
    X_gp, Y_gp, X_proc, Y_proc, gp_list = group_distribute(X_gp, Y_gp, ngroups_all)
    # the initial best tree on each process only have z for data on each process
    if !rand
        z_proc = bound.warm_start_z(X_proc, warm_start.a, warm_start.b, D)
        UB_tree = Tree(warm_start.a, warm_start.b, warm_start.c, warm_start.d, z_proc, D)
    else # rand data generation for ub selection does not need to update best tree
        UB_tree = nothing
    end
    # length of samples on each process
    if length(X_proc) == 0
        n = 0
    else
        n = size(X_proc)[2] # mapreduce(x->length(x), +, groups_all[gp_list]) # length(reduce(vcat, groups_all[gp_list]))
    end
    ngroups = length(gp_list)
    # get group index for the sub-dataset in each process
    groups = groups_on_proc(groups_all, gp_list)
    # get group_trees
    group_trees = group_trees_init(warm_start, ngroups)
    # add dummy group info to balance each core, used in UB computation
    max_l_len = parallel.get_max_list_length()
    if ngroups < max_l_len
        for i in 1:max_l_len-ngroups
            push!(groups, 0:-1) # dummy group with length(group) = 0
            push!(group_trees, Tree()) # dummy tree with tree.D = 0
        end
    end
    LB_gp = zeros(max_l_len)
    lrg_gap = falses(max_l_len)
    lower, upper = bound.init_bound(p, n, K, D, nothing, nothing, val) # lower and upper are in Tree type # here z only for data of each process
    # this node is the root branching node, node level start from 1 for debugging and change back to 0 after debug
    node = Node(lower, upper, 1, -1e15, -ones(n), groups, nothing, group_trees, LB_gp, lrg_gap, "d"); 
    return X_gp, Y_gp, X_proc, Y_proc, node, UB_tree
end



end