module bb_func

#using Clustering#, Metaheuristics
using Printf
using JuMP
using Random, LinearAlgebra, SparseArrays
using Statistics, StatsBase, Distances
using MPI
using Distributed, SharedArrays

@everywhere using Trees, Nodes, branch, bound, parallel, opt_func, ub_func, lb_func, groups

export branch_bound

maxiter = 100000000000
tol = 1e-6
#time_lapse = 600*6*4 # 1*3600 # 4 hours

# function to record the finish time point
time_finish(seconds) = round(Int, 10^9 * seconds + time_ns())

# during iteration: 
# LB = node.LB is the best LB among all node and is in current iteration
# UB is the best UB, and node_UB the updated UB(it is possible: node_UB > UB) of current iteration (after run getUpperBound)
# node_LB is the updated LB of current itattion (after run probing or getLowerBound_adptGp_LD)
function branch_bound(X, Y, K, D, warm_start, UB_init, alpha, L_hat, method = "CF", prob = false, obbt = false, val=0, time_lapse::Int64=14400)
    # parameter initialization
    if parallel.is_root()
        p, n_all = size(X);
        alpha_s = alpha/n_all;
        sortX = sort(X, dims=2) # sorted on each feature used in lb_func
        eps = vec(mapslices(opt_func.mini_dist, sortX, dims=2)) # eps used in opt_func
    else
        sortX = nothing # not used in parallel and no need for broadcasting
        p = nothing
        n_all = nothing
        alpha_s = nothing
        eps = nothing
    end
    p = parallel.bcast(p)
    n_all = parallel.bcast(n_all)
    alpha_s = parallel.bcast(alpha_s)
    eps = parallel.bcast(eps)

    # All_proc Initialization
    UB = UB_init;
    max_LB = 1e15; # used to save the best lower bound at the end (smallest but within the mingap)
    # distribute data to each process and generate corresponding root_node for each process
    ~, ~, X_proc, Y_proc, node, tree = groups.proc_data_preparation(X, Y, p, n_all, K, D, warm_start, method, false, 0, val)
    # generate random subsets for UB computation
    X_rand, Y_rand, X_rproc, Y_rproc, node_rand, ~ = groups.proc_data_preparation(X, Y, p, n_all, K, D, warm_start, method, true)

    # start bound calculation for root node
    iter = 0
    node, UB, tree, fathom = getBound(X_proc, Y_proc, X_rproc, Y_rproc, node, node_rand, K, D, eps, UB, tree, alpha_s, L_hat, mingap, method, iter, false)
    # each nodeList stores the node on dataset of each process
    nodeList =[]
    LB_list = []
    push!(nodeList, node)
    push!(LB_list, node.LB)
    # parameters are store in root process
    if parallel.is_root()
        println("Iter\tleft\tlev\tLB\tUB\tgap\t")
    end
    # get program end time point
    end_time = time_finish(time_lapse) # the branch and bound process ends after 6 hours
    #####inside main loop##################################
    calcInfo = [] # initial space to save calcuation information
    
    while nodeList != []
        node = nodeList[1] # node = nodeList[nodeid] # sorted node list, the first has the lowest LB
        LB = LB_list[1] # node_LB is current lowerest LB
        deleteat!(nodeList, 1)#deleteat!(nodeList, nodeid) # delete the to-be-processed node
        deleteat!(LB_list, 1) # delete the lowerest lb
        # so currently, the global lower bound corresponding to node, LB = node.LB, groups = node.groups
        if parallel.is_root()
            @printf "%-6d %-6d %-10d %-10.4f %-10.4f %-10.4f %s \n" iter length(nodeList) node.level LB UB (UB-LB)/min(abs(LB), abs(UB))*100 "%"
        end
        # time stamp should be checked after the retrival of the results
        if (iter >= maxiter) || (time_ns() >= end_time)
            if parallel.is_root()
                push!(calcInfo, [iter, length(nodeList), node.level, LB, UB, (UB-LB)/min(abs(LB), abs(UB))])
            end
            break
        end
        iter += 1
        
        ############# iteratively bound tightening #######################
        # the following code delete branch with lb close to the global upper bound
        delete_nodes = []
        for (idx,n) in enumerate(nodeList)
            if (((UB-n.LB)<= mingap) || ((UB-n.LB) <=mingap*min(abs(UB), abs(n.LB))))
                push!(delete_nodes, idx)
            end
        end
        deleteat!(nodeList, sort(delete_nodes))
        deleteat!(LB_list, sort(delete_nodes))
        ##################### lower and upper(inside lb function) bound update #####################
        X_rand, Y_rand, X_rproc, Y_rproc, node_rand, ~ = groups.proc_data_preparation(X, Y, p, n_all, K, D, warm_start, method, true, iter)
        node, UB, tree, fathom = getBound(X_proc, Y_proc, X_rproc, Y_rproc, node, node_rand, K, D, eps, UB, tree, alpha_s, L_hat, mingap, method, iter, false) 

        ##################### branching #####################
        if fathom
            # save the best LB if it close to UB enough (within the mingap)
            parallel.root_println("LB close or above UB, fathomed")
            if node.LB < max_LB && node.LB <= UB+1e-10
                max_LB = node.LB
            end
            # continue   
        else
            if parallel.is_root() # generate branck info on root node
                bVarIdx, bVar = branch.SelectVarSequential(node,D)
                println("branching on $bVarIdx with $bVar.")
            else
                bVarIdx = nothing
                bVar = nothing
            end
            # broadcasting
            bVarIdx = parallel.bcast(bVarIdx)
            bVar = parallel.bcast(bVar)
            if bVar != "stop" # means we can continue branching
                if bVar == "b"
                    # the split value is chosen by the midpoint
                    bValue = (node.upper.b[bVarIdx] + node.lower.b[bVarIdx])/2;
                else
                    bValue = nothing
                end
                branch!(nodeList, LB_list, bVar, bVarIdx, bValue, node, sortX)
            end
        end
    end
    
    if parallel.is_root()
        if nodeList==[]
            println("all node solved")
            push!(calcInfo, [iter, length(nodeList), max_LB, UB, (UB-max_LB)/min(abs(max_LB), abs(UB))])
        else
            max_LB = calcInfo[end][4]
        end
        println("solved nodes:  ", iter)
        @printf "%-52d  %-14.4e %-14.4e %-7.4f %s \n" iter  max_LB UB (UB-max_LB)/min(abs(max_LB),abs(UB))*100 "%"
    end
    parallel.barrier()
    return tree, UB, calcInfo, max_LB
end

end