module branch

using Random
using Nodes, Trees, bound, parallel

export branch!

function sltVar_b(lb, ub, D)
    weight = ones(2^D-1)
    for i=1:D
        weight[2^(i-1):(2^i-1)] .= D-i+1
    end
    dif = weight.*(ub-lb)
    ind = findmax(dif)[2] # return idx of Tb node for b
    return ind, "b"
end

function SelectVarSequential(node, D)
    # idx determine the precedence of branching 
    lwr = node.lower
    upr = node.upper
    bch_var = node.bch_var
    # first branch d
    d_udt, d_dt = bound.bound_idx(lwr.d, upr.d)
    if length(d_udt) >= 1
        return d_udt[1], "d"
    end
    #seed = Int(round(maximum(abs.(upr.b-lwr.b))))
    #Random.seed!(seed)
    # for parallel, rand() will be different on each processor, to main the same make sure the same rand() broadcast to all processors
    token = rand() 
    threshold =  1-0.5*maximum(abs.(upr.b-lwr.b))
    #parallel.root_println("threshold:  $threshold")
    if token <= threshold
        # then branch a
        a_udt, a_dt = bound.bound_idx(lwr.a, upr.a)
        if length(a_udt) >= 1
            return a_udt[1], "a"
        end
        c_udt, c_dt = bound.bound_idx(lwr.c, upr.c)
        if length(c_udt) >= 1
            return c_udt[1], "c"
        end
    end
    return sltVar_b(lwr.b, upr.b, D)
end


function branch!(nodeList, LB_list, bVar, bVarIdx, bValue, node, sortX::Union{Nothing, Matrix{Float64}})
    insert_id = searchsortedfirst(LB_list, node.LB) # get the position to insert the current split nodes

    lower = Trees.copy_tree(node.lower)
    upper = Trees.copy_tree(node.upper)
    bound.update_bound!(lower, upper, bVar, bVarIdx, bValue, "right") # split from this variable at bValue
    if parallel.is_root()
        fathom_r = bound.check_bound_b(lower, upper, sortX) # if fathom, this branch will not saved
    else
        fathom_r = nothing
    end
    fathom_r = parallel.bcast(fathom_r)

    if sum(lower.a.>upper.a)==0 && sum(lower.b.>upper.b)==0 &&
       sum(lower.c.>upper.c)==0 && sum(lower.d.>upper.d)==0 && !fathom_r
        # node.level*2+1 is the index of the new right node instead of node.level+1 change back after debug
        right_node = Node(lower, upper, node.level+1, node.LB, copy(node.costs), node.groups, nothing, copy(node.group_trees), copy(node.LB_gp), copy(node.lrg_gap), bVar)
        # push!(nodeList, left_node)
        insert!(nodeList, insert_id, right_node)
        insert!(LB_list, insert_id, node.LB)
        # println("left_node:   ", lower, "   ",upper)
    end

    lower = Trees.copy_tree(node.lower)
    upper = Trees.copy_tree(node.upper)
    bound.update_bound!(lower, upper, bVar, bVarIdx, bValue, "left") # split from this variable at bValue
    if parallel.is_root()
        fathom_l = bound.check_bound_b(lower, upper, sortX) # if fathom, this branch will not saved
    else
        fathom_l = nothing
    end
    fathom_l = parallel.bcast(fathom_l)
    
    if sum(lower.a.>upper.a)==0 && sum(lower.b.>upper.b)==0 && 
       sum(lower.c.>upper.c)==0 && sum(lower.d.>upper.d)==0 && !fathom_l
        # node.level*2 is the index of the new left node instead of node.level+1 change back after debug
    	left_node = Node(lower, upper, node.level+1, node.LB, copy(node.costs), node.groups, nothing, copy(node.group_trees), copy(node.LB_gp), copy(node.lrg_gap), bVar)
	    # push!(nodeList, left_node)
	    insert!(nodeList, insert_id, left_node)
        insert!(LB_list, insert_id, node.LB)
        # println("left_node:   ", lower, "   ",upper)
    end

end

end