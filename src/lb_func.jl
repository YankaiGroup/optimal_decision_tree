module lb_func

using Random, Distributions
using LinearAlgebra, SparseArrays, Statistics, StatsBase
using MLDataUtils, Clustering
using JuMP
using TimerOutputs: @timeit, get_timer

using Distributed, SharedArrays
using parallel
@everywhere using ub_func, opt_func, Trees, Nodes, bound

export getBound


tol = 1e-6
max_iter = 1000


################### Closed-form lower bound calculation ###################
function get_direct(xs, lower, upper, a_udt, d_dt, t, Tl)
    t_idx = []
    if t in d_dt && lower.d[t] == 0 # only when t is determined and d[t] = 0 will have sure no split
        while t < Tl # all decesendent is right directed to the leaf with d=0
            t = 2*t+1
        end
        push!(t_idx, t)
    else # default d[t] == 1
        ar_uid = findall(x->x[2]==t, a_udt) # get idx of undetermined a on col t
        if isempty(ar_uid) # a[t] is fixed on particular feature
            fea = findall(x->x==1, lower.a[:,t])[1]
            if xs[fea] < lower.b[t] # case1 s[f] < lower.b[t], then go left t <- 2t
                push!(t_idx, 2*t)
            elseif xs[fea] >= upper.b[t] # case2 s[f] >= upper.b[t], then go right, t <- 2t+1
                push!(t_idx, 2*t+1)
            else # case3 s[f]∈[lower.b[t], upper.b[t]), then both, t <- 2t and t <- 2t+1
                push!(t_idx, 2*t, 2*t+1)
            end
        else
            #println("adt on some fea at node $t")
            carId_udt = a_udt[ar_uid] # get the CartesianIndex of col t on a (for undetermined)
            for i in carId_udt::Vector{CartesianIndex} # for each undetermined feature of a on col t
                fea = i[1]
                if xs[fea] < lower.b[t] # case1 s[f] < lower.b[t], then go left t <- 2t
                    2*t in t_idx ? nothing : push!(t_idx, 2*t)
                elseif xs[fea] >= upper.b[t] # case2 s[f] >= upper.b[t], then go right, t <- 2t+1
                    2*t+1 in t_idx ? nothing : push!(t_idx, 2*t+1)
                else # case3 s[f]∈[lower.b[t], upper.b[t]), then both, t <- 2t and t <- 2t+1
                    2*t in t_idx ? nothing : push!(t_idx, 2*t)
                    2*t+1 in t_idx ? nothing : push!(t_idx, 2*t+1)
                end
            end
        end
    end
    return t_idx
end


function check_CF(xs, ys, lower, upper, Tl)
    t = 1
    lidx = findall(x->x==1, ys)[1] # get the true label of the sample 
    z = zeros(Tl)
    cost_leaf = (Tl+1)*ones(Tl)
    lb_list = ones(Tl) # initial lb for each leaf as one, if sample can reach and label match, update to zero
    nodelist = Int64[]
    push!(nodelist, t) # push first node idx in to list
    while nodelist != []
        t = popfirst!(nodelist) # get current node idx t which is at the first of the Array
        if t < Tl
            if upper.d[t] == 0
                while t < Tl
                    t = 2*t+1
                end
                push!(nodelist, t) 
            else
                fset_t = findall(x->x==1, upper.a[:,t])
                chk_l = xs[fset_t] .< lower.b[t] # check whether the sample is smaller than the lower bound of the node t
                if sum(chk_l) == length(chk_l) # if all the feature value is smaller than the lower bound of the node t
                    push!(nodelist, 2*t)
                elseif sum(chk_l) == 0 # all the feature value is larger than the lower bound of the node t
                    push!(nodelist, 2*t+1)
                    chk_u = xs[fset_t] .>= upper.b[t] # check whether the sample is larger than the upper bound of the node t
                    if sum(chk_u) < length(chk_u) # if not all the feature value is larger than the upper bound of the node t
                        push!(nodelist, 2*t)
                    end 
                else # some feature value is smaller than the lower bound and some is larger than the upper bound
                    push!(nodelist, 2*t)
                    push!(nodelist, 2*t+1)
                end
            end
        else # t in leaf
            t_i = t-Tl+1 # since here c and z only have Tl elements, idx t should be transferred
            z[t_i] = 1 # this leaf can be reached by the sample s
            # check whether the true label idx is determined and true label (should has value 1) is not equal to the dtm value
            if upper.c[lidx, t] == 0 # && CartesianIndex(lidx, t) in c_dt
                lb_list[t_i] = 1 # cost 1
                cost_leaf[t_i] = 1
            elseif lower.c[lidx, t] == 1
                lb_list[t_i] = 0
                cost_leaf[t_i] = 0
            else
                # other condition, c[lidx, t] is undetermined
                lb_list[t_i] = 0
            end
        end 
    end
    # get reached leaf index
    z_udt = findall(x->x==1, z) 
    if sum(cost_leaf[z_udt]) == 0
        cost = 0
    elseif sum(cost_leaf[z_udt]) == length(z_udt)
        cost = 1
    else # some leaf has lb 0 and some leaf has lb 1
        cost = -1
    end
    LB = minimum(lb_list[z_udt]) # if all leaf has lb=1, then LB of xs is 1 else 0.
    return z_udt, LB, cost
end

function CF(X_cf, Y_cf, K, D, lower, upper, dtm_idx, costs, alpha_s, L_hat)
    if length(X_cf) == 0
        return 0, lower, upper, dtm_idx, costs
    end
    p, n = size(X_cf)
    Tb = 2^D-1
    Tl = 2^D
    T = Tb+Tl
    if lower === nothing || upper === nothing
        lower, upper = bound.init_bound(p, n, K, D)
    end
    # here z_udt can be dtm_idx[7] and pass to CF to reduce the calculation load
    z_udt = Vector{Int64}[]
    LB = 0 # for each rank
    i = 1
    for s in 1:n::Int64
        if costs[i] == -1 # cost of s is not determined
            xs = X_cf[:,s]
            ys = Y_cf[:,s]
            z_udt_s, LB_s, cost = check_CF(xs, ys, lower, upper, Tl)
            # get the idx of z that sample will reach, won't have 0 element since must have one leaf can be reached
            costs[i] = cost 
            bound.update_zs!(lower, upper, i, z_udt_s)
        else # cost of s is determined
            z_udt_s = Int64[] # only when cost of s is determined then z[s] can be empty, other must has one leaf to be reached
            LB_s = costs[i] # same as costs[s]
        end
        push!(z_udt, z_udt_s)
        LB += LB_s
        i += 1
    end    
    dropzeros!(upper.z) # remove zero value on sparse matrix
    upr_I, upr_J, upr_V = findnz(upper.z)
    upper = Tree(upper.a, upper.b, upper.c, upper.d, sparse(upr_I, upr_J, upr_V, n, Tl), D) # no need to update lwr
    dtm_idx[7] = z_udt
    # costs = -ones(n) # close sample reduction
    return 1/L_hat*LB+alpha_s*n*sum(lower.d), lower, upper, dtm_idx, costs
end

################### grouping lower and upper bound calculation ###################
function SG_solver(X_proc, Y_proc, K, D, alpha, L_hat, group, lower, upper, eps, dtm_idx, tree, LB_gp_old, lrg_gap, costs, costs_udt, costs_dt, iter, rlx, time_limit)
    if sum(tree.a .< lower.a)==0 && sum(tree.a .> upper.a)==0 && 
        sum(tree.b .< lower.b)==0 && sum(tree.b .> upper.b)==0 && 
        sum(tree.c .< lower.c)==0 && sum(tree.c .> upper.c)==0 && 
        sum(tree.d .< lower.d)==0 && sum(tree.d .> upper.d)==0 && 
        !lrg_gap && iter > 0
        objv = LB_gp_old
        new_tree_gp = false
    else
        idx = filter(x->x in costs_udt, group) # get index of sample(for whole dataset) that is in costs_udt from groups[i]
        idx_dt = filter(x->x in costs_dt, group) # get index of sample with determined cost
        lwr_i = Tree(lower.a, lower.b, lower.c, lower.d, lower.z[idx,:], D)
        upr_i = Tree(upper.a, upper.b, upper.c, upper.d, upper.z[idx,:], D)
        dtm_idx_i = vcat(dtm_idx[1:6], [dtm_idx[7][idx]])
        # since tree.z contains for each subproblem sample, thus we should get the index of sample on group[i] for each subproblem
        gp_idx = findall(x->x in idx, group) # get the index of the selected sample(labeled in idx) from groups[i]
        new_z = bound.warm_start_z(X_proc[:,gp_idx], tree.a, tree.b, D)
        ws_i = Tree(tree.a, tree.b, tree.c, tree.d, new_z, D)
        mute = true
        tree, objv, gap = opt_func.global_OPT_DT_SG(X_proc[:,idx], Y_proc[:,idx], K, D, alpha*length(group), L_hat; lower=lwr_i, upper=upr_i, eps=eps, dtm_idx=dtm_idx_i, w_sos=nothing, lambda = nothing, warm_start = ws_i, mute=mute, rlx=rlx, time=time_limit)
        if gap > mingap
            lrg_gap = true
        end
        # add determined costs
        objv += 1/L_hat*sum(costs[idx_dt])
        new_tree_gp = true
    end
    return tree, objv, lrg_gap, new_tree_gp
end

function SG(X_lb, Y_lb, X_ub, Y_ub, K, D, ws_trees, groups, lower, upper, eps, dtm_idx, costs, UB_old, tree_old, LB_gp_old, lrg_gap_old, alpha_s, L_hat, iter = 0, rlx = false, time_limit = 60)
    if length(X_lb) == 0
        p = 0
        n_lb = 0
        n_ub = 0
    else
        p,n_lb = size(X_lb)
        n_ub = size(X_ub)[2] # when SG in lb_calc, n_ub=n for rand ub select, n_ub is the size of group data in each process
    end
    ngroups = length(groups)
    # lower bound Initialization
    LB = 0
    LB_gp = Array{Float64}(undef, ngroups)
    lrg_gap = falses(ngroups)
    trees_gp = [Tree(p,D,n_lb,K) for i in 1:ngroups] 
    costs_udt = findall(x->x==-1, costs) # index on all sample
    costs_dt = findall(x->x!=-1, costs)
    # upper bound Initialization
    UB = UB_old #change to UB global from input
    tree = Trees.copy_tree(tree_old)
    z_pos = findnz(tree_old.z)[2]
    # start bound calculation
    for i in 1:ngroups::Int64
        # group process        
        group = groups[i]
        ws_tree = ws_trees[i]
        LB_gp_old_i = LB_gp_old[i]
        lrg_gap_old_i = lrg_gap_old[i]
        # start lower bound calculation
        if length(group) > 0
            tree_i, LB_i, lrg_gap_i, new_tree_gp_i = SG_solver(X_lb, Y_lb, K, D, alpha_s, L_hat, group, lower, upper, eps, dtm_idx, ws_tree, LB_gp_old_i, lrg_gap_old_i, costs, costs_udt, costs_dt, iter, rlx, time_limit)
            trees_gp[i] = tree_i; # i for groups and centers are corresponding to i+1 of lambda and trees is the tree parameters in type of Tree
            LB_gp[i] = LB_i # update cuts info: opt lb value
            lrg_gap[i] = lrg_gap_i
            LB += LB_i; 
        else # used for dummy group
            tree_i = Tree()
            trees_gp[i] = tree_i; # if no update, tree set to empty. tree.D = 0
            LB_gp[i] = LB_gp_old_i
            lrg_gap[i] = lrg_gap_old_i
        end
        parallel.barrier()
        # start upper bound calculation
        UB, tree, z_pos = ub_func.distributed_UB(tree_i, UB, tree, z_pos, X_ub, Y_ub, n_ub, alpha_s, L_hat)
    end
    z = sparse(1:n_ub, z_pos, ones(n_ub), n_ub, 2^D)
    tree = Tree(tree.a, tree.b, tree.c, tree.d, z, D) # update tree with z, z only store value for samples at each process
    return LB, LB_gp, lrg_gap, trees_gp, UB, tree
end


function lb_calc(X_proc, Y_proc, K, D, lower, upper, dtm_idx, costs, group_trees, groups, UB, tree, eps, LB_gp, lrg_gap, alpha_s, L_hat, mingap, LB_mtd, iter, updateUB = false, LB_small_test=true)
    fathom = false
    ##### first level #####
    # using CF to determine the z value   
    if "CF" in LB_mtd
        LB, lower, upper, dtm_idx, costs = CF(X_proc, Y_proc, K, D, lower, upper, dtm_idx, costs, alpha_s, L_hat)
        LB = parallel.sum(LB) # sum the lower bound of all cores
        if (UB-LB)<= mingap || (UB-LB) <= mingap*min(abs(LB), abs(UB))
            fathom = true
        end
        parallel.barrier()
        ##### second level #####
        # LB can be obtained through optimizer with only c under the determination of leaf-reach(z) from CF
        if !fathom && "MILP" in LB_mtd && lower.c != upper.c # if lower.c == upper.c, no need to run MILP
            # first get cost udt and transmit udt base info to root, then reduce cost dt and add to LB
            costs_udt = findall(x->x==-1, costs)
            costs_dt = findall(x->x!=-1, costs)
            # var to transmit are X_gp, Y_gp, lz, uz, tz, dtmidx z, and reduce costdt lb, the hard thing is lz, uz，tz all need to be vector and the idx are all start from one
            X_milp = parallel.collect(X_proc[:, costs_udt])
            Y_milp = parallel.collect(Y_proc[:, costs_udt])
            dtmz_all = parallel.collect(dtm_idx[7][costs_udt])
            LB_dt = parallel.sum(sum(costs[costs_dt]))

            if parallel.is_root()
                n_all = size(X_milp)[2]
                lz_all = sparse(Int64[], Int64[], Float64[], n_all, 2^D)
                uz_I = Int64[]
                for i in 1:n_all
                    append!(uz_I, i*ones(length(dtmz_all[i])))
                end
                uz_J = Int64[]
                for i in 1:n_all
                    append!(uz_J, dtmz_all[i])
                end
                uz_V = ones(length(uz_I))
                uz_all = sparse(uz_I, uz_J, uz_V, n_all, 2^D)
                lwr = Tree(lower.a, lower.b, lower.c, lower.d, lz_all, D)
                upr = Tree(upper.a, upper.b, upper.c, upper.d, uz_all, D)
                # tree.z are got from calculation of current optimal tree
                tz_all = bound.warm_start_z(X_milp, tree.a, tree.b, D)
                ws_tree = Tree(tree.a, tree.b, tree.c, tree.d, tz_all, D)
                # here X, Y are the global data from the root.
                c, LB_MILP = opt_func.global_OPT_DT_MILP(X_milp, Y_milp, K, D, alpha_s*n_all, L_hat; lower=lwr, upper=upr, dtm_idx=vcat(dtm_idx[1:6], [dtmz_all]), warm_start=ws_tree, mute=true)
                LB_MILP += 1/L_hat*LB_dt
                if LB_MILP > LB
                    LB = LB_MILP
                    if (UB-LB)<= mingap || (UB-LB) <= mingap*min(abs(LB), abs(UB))
                        fathom = true
                    end
                end
            end
            LB = parallel.bcast(LB)
            fathom = parallel.bcast(fathom)
        end
        ##### third level ##### # can also be launched alone
        if !fathom && "SG" in LB_mtd # LB obtained through grouping
            if LB_small_test && parallel.nprocs() == 1 ### add costs info
                ngroups = length(groups)
                ~, n = size(X);
                n_sub = 5
                obj_trial = 0### change to mpi using SG_solver
                for t = 1:n_sub
                    #Random.seed!(1)
                    i = rand(1:ngroups)
                    lwr_i = Tree(lower.a, lower.b, lower.c, lower.d, lower.z[groups[i],:], D)
                    upr_i = Tree(upper.a, upper.b, upper.c, upper.d, upper.z[groups[i],:], D)
                    dtm = bound.boundIdx_all(lwr_i, upr_i, Vector{Int64}[])
                    ws_i = group_trees[i]
                    new_z = bound.warm_start_z(X[:,groups[i]], ws_i.a, ws_i.b, D)
                    ws_i = Tree(ws_i.a, ws_i.b, ws_i.c, ws_i.d, new_z, D)
                    # get the idx for z that sample can reach
                    ~, objv,~ = opt_func.global_OPT_DT_SG(X_proc[:,groups[i]], Y_proc[:,groups[i]], K, D, alpha_s, L_hat;lower=lwr_i, upper=upr_i, eps=eps, dtm_idx=dtm, w_sos=nothing, lambda = nothing, warm_start = ws_i, mute=true, rlx=false)
                    obj_trial += objv
                end
                if obj_trial >= UB
                    return obj_trial, lower, upper, costs, LB_gp, lrg_gap, group_trees, fathom, UB, tree
                end
            end
            @timeit get_timer("Shared") "Bound Calculation (LB and UB1) " begin
            LB_SG, LB_gp, lrg_gap, group_trees, UB, tree = SG(X_proc, Y_proc, X_proc, Y_proc, K, D, group_trees, groups, lower, upper, eps, dtm_idx, costs, UB, tree, LB_gp, lrg_gap, alpha_s, L_hat, iter, false, 60*1)   # run 3 mins
            LB_SG = parallel.sum(LB_SG) # sum the lower bound of all cores
            if LB_SG > LB
                LB = LB_SG
                if (UB-LB)<= mingap || (UB-LB) <= mingap*min(abs(LB), abs(UB))
                    fathom = true
                end
            end
            end
        end
    end
    return LB, lower, upper, costs, LB_gp, lrg_gap, group_trees, fathom, UB, tree
end


function getBound(X_proc, Y_proc, X_rproc, Y_rproc, node, node_rand, K, D, eps, UB, UB_tree, alpha_s, L_hat, mingap, LB_mtd = "SG", iter = 0, LB_small_test=true)
    lower = node.lower
    upper = node.upper
    costs = node.costs
    LB_gp = node.LB_gp
    lrg_gap = node.lrg_gap
    # check the bound of variables # Vector{Int64}[] is init for z_udt
    dtm_idx = bound.boundIdx_all(lower, upper, Vector{Int64}[])
    LB, lower, upper, costs, LB_gp, lrg_gap, group_trees, fathom, UB, UB_tree = lb_calc(X_proc, Y_proc, K, D, lower, upper, dtm_idx, costs, node.group_trees, node.groups, UB, UB_tree, eps, LB_gp, lrg_gap, alpha_s, L_hat, mingap, LB_mtd, iter, true, LB_small_test)
    LB = max(node.LB, LB)
    # update best solution and objective_value 
    @timeit get_timer("Shared") "UB2" begin   
        if (UB-LB)> mingap && (UB-LB) > mingap*min(abs(LB), abs(UB))
            #println("rank: $(parallel.myid()), UB: $UB")
            if lower.a == upper.a && lower.d == upper.d && lower.c == upper.c
                # [lower, upper] is a vector and input as a vector the element can be changed even in the function
                trees_ub2 = parallel.nprocs() <= 3 ? [lower, upper] : [lower]
                UB_tree, UB = ub_func.UB_select(trees_ub2, UB, UB_tree, X_proc, Y_proc, K, D, alpha_s, L_hat, nothing, lower.b, upper.b)
            end
            if "SG" in LB_mtd 
                # update UB with bootstrapped data
                ############# the no solution bug is the problem of z, costs and all that related to X_proc #############
                dtm_idx_rand = bound.boundIdx_all(node_rand.lower, node_rand.upper, Vector{Int64}[])
                ~, lower_rand, upper_rand, dtm_idx_rand, costs_rand = CF(X_rproc, Y_rproc, K, D, node_rand.lower, node_rand.upper, dtm_idx_rand, node_rand.costs, alpha_s, L_hat)
                ~, ~, ~, ~, UB, UB_tree = SG(X_rproc, Y_rproc, X_proc, Y_proc, K, D, node_rand.group_trees, node_rand.groups, lower_rand, upper_rand, eps, dtm_idx_rand, costs_rand, UB, UB_tree, node_rand.LB_gp, node_rand.lrg_gap, alpha_s, L_hat, iter, false, 60) # ub calc run only 45 secs
            else      
                node_tree, node_UB = ub_func.getUpperBound(X_proc, Y_proc, K, D, alpha_s, L_hat, UB, UB_tree, "heur", group_trees, lower, upper)
                if (node_UB < UB)
                    UB = node_UB
                    UB_tree = node_tree
                end 
            end
            parallel.root_println("UB: $UB")     
        end
    end
    
    #GC.gc() 
    return Node(lower, upper, node.level, LB, costs, node.groups, node.lambda, group_trees, LB_gp, lrg_gap, node.bch_var), UB, UB_tree, fathom
end


end