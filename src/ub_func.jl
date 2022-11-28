module ub_func

using DecisionTree, StatsBase
using Printf
using JuMP
using Random, SparseArrays
using Trees, bound, parallel
using opt_func

export getUpperBound, CART_base, UB_update, predict_oct



function getUpperBound(X, Y, K, D, alpha, L_hat, UB, UB_tree, method = "heur", trees=nothing, lower=nothing, upper=nothing, new_tree_gp = nothing)
    if method == "heur"
        tree, objv = CART_base(X, Y, K, D, alpha, L_hat, lower=lower, upper=upper)
    elseif method == "merg"
        tree, n_wrong = UB_update(trees, X, Y, K, D)
        objv = 1/L_hat*n_wrong+sum(tree.d)*alpha
    else # "slt"
        tree, objv = UB_select(trees, UB, UB_tree, X, Y, K, D, alpha, L_hat, new_tree_gp)
    end
    return tree, objv
end

function distributed_UB(tree_i, UB, tree, z_pos, X_proc, Y_proc, n, alpha_s, L_hat)
    #println(tree)
    #println(z_pos)
    tree_list = parallel.allcollect([tree_i])
    UB_list = Float64[]
    z_pos_list = Vector{Int64}[]
    for t_i in tree_list::Vector{Tree}
        if t_i.D != 0 
            UB_gp_i, z_pos_gp_i = ub_func.objv_cost(X_proc, Y_proc, n, t_i, alpha_s*n, L_hat) # here X, Y are global data and label.
            # upper bound info gathering
            push!(UB_list, UB_gp_i)
            push!(z_pos_list, z_pos_gp_i)
        else # only dummy tree has D = 0 and will not be used in UB calculation, set UB to Inf so that not to be selected
            push!(UB_list, Inf)
            push!(z_pos_list, [-1])
        end
    end
    parallel.barrier()
    # allreduce the ub results and get the best(minimum) UB
    parallel.sumv!(UB_list)
    UB_i = minimum(UB_list)
    proc_idx = argmin(UB_list)
    # update the UB, tree and z_pos 
    if UB_i < UB
        UB = UB_i
        tree = tree_list[proc_idx]
        z_pos = z_pos_list[proc_idx]
    end
    parallel.barrier()
    return UB, tree, z_pos
end

function UB_select(trees, UB_old, UB_tree_old, X_proc, Y_proc, K, D, alpha_s, L_hat, new_tree_gp = nothing, lwr_b=nothing, upr_b=nothing)
    p,n = size(X_proc)
    Tb = 2^D-1
    n_trees = length(trees)
    if new_tree_gp === nothing
        new_tree_gp = trues(n_trees)
    end
    UB = UB_old
    tree = Trees.copy_tree(UB_tree_old)
    z_pos = findnz(UB_tree_old.z)[2]
    for i in 1:n_trees::Int
        if new_tree_gp[i]
            tree_i = Trees.copy_tree(trees[i])
            if lwr_b !== nothing
                Random.seed!(i*(parallel.myid()+1))
                tree_i.b[:] = rand(Tb).*(upr_b .- lwr_b) .+ lwr_b 
            end
            UB, tree, z_pos = distributed_UB(tree_i, UB, tree, z_pos, X_proc, Y_proc, n, alpha_s, L_hat)
        end
    end
    
    z = sparse(1:n, z_pos, ones(n))
    tree = Tree(tree.a, tree.b, tree.c, tree.d, z, tree.D)
    return tree, UB
end


function CART_base(X, Y, K, D, alpha, L_hat; lower=nothing, upper=nothing, prune_val=1.0)
    y = opt_func.label_int(Y)
    tree = warm_start(X, y, K, D, prune_val)
    accr, pred, z_pos = predict_oct(X, y, tree.a, tree.b, round.(tree.c))
    objv = 1/L_hat*(1-accr)*length(y) + alpha*sum(tree.d)
    return tree, objv
end


# in warm start the output structure is exactly the same as dimitris and glb_DT, including structure of c
function warm_start(X, y, K, D = 4, prune_val=0.0)
    p,n = size(X)
    Tb = 2^D-1
    T = 2^(D+1)-1
    Random.seed!(1)
    # for original cart version (use package DecisionTree/v0.10.12)
    cart_model = DecisionTreeClassifier(pruning_purity_threshold=prune_val, max_depth=D, min_samples_leaf = ceil(0.05*n)) # 
    DecisionTree.fit!(cart_model, X', y)
    a,b,c,d = warm_start_params(zeros(p,Tb), zeros(Tb), zeros(T), zeros(Tb), 1, cart_model.root, 2^D:T)
    
    z = bound.warm_start_z(X, a, b, D)
    c_bin = zeros(K,T)
    for i in 1:T
        k = Int(c[i])
        if k != 0
            c_bin[k,i] = 1
        end
    end
    return Tree(a,b,c_bin,d,z,D)
end


function warm_start_params(a,b,c,d,t,node,Tl)
    if node isa Leaf
        t_leaf = t
        while !(t_leaf in Tl)
            t_leaf = 2*t_leaf+1
        end
        c[t_leaf] = node.majority
    else
        a[node.featid, t] = 1
        b[t] = node.featval
        d[t] = 1
        a,b,c,d = warm_start_params(a,b,c,d,2*t, node.left, Tl)
        a,b,c,d = warm_start_params(a,b,c,d,2*t+1, node.right, Tl)
    end
    return a,b,c,d
end

# prune tree through merge samples under split dt node into one leaf
function tree_prune(X::Matrix{Float64}, tree_orig::Tree, val::Int64=0)
    tree = Trees.copy_tree(tree_orig)
    Tb = 2^tree.D-1
    if val == 0 # if val == 0, it is not in validation mode
        d = nothing 
    else # d will be a vector converted from integer
        d = float(reverse(digits(val, base=2, pad=Tb)))
    end
    if d === nothing 
        return tree
    else
        t= 1
        nodelist = []
        push!(nodelist, t)
        while nodelist != []
            t = popfirst!(nodelist)
            if t <= Tb
                if d[t] == 0.0
                    list_b = Int64[]
                    list_l = Int64[]
                    # descendants of bVarIdx and itself are all set to zero
                    bound.getchild!(list_b, list_l, t, Tb)
                    tree.d[list_b] .= 0.0
                    tree.a[:,list_b] .= 0.0
                    tree.b[list_b] .= 0.0
                    t = pop!(list_l) # the last leaf (right most) has label
                    tree.c[:,list_l] .= 0.0
                    push!(nodelist, t)
                else
                    if tree.d[t] == 0.0
                        tree.d[t] = 1.0
                        tree.a[1,t] = 1.0
                        tree.b[t] = 0.5
                    end
                    push!(nodelist, 2*t)
                    push!(nodelist, 2*t+1)
                end
            else
                if sum(tree.c[:,t]) == 0.0
                    tree.c[1,t] = 1.0
                end
            end
        end
        z = bound.warm_start_z(X, tree.a, tree.b, tree.D)
        return Tree(tree.a, tree.b, tree.c, tree.d, z, tree.D)
    end
end

function UB_update(trees, X, y, k, D)
    ngroups = length(trees)
    p, n = size(X)
    Tb = 2^D-1
    Tl = 2^D
    T = Tb+Tl
    # first, determine node split from d
    d = round.(sum(map(p->p.d, trees))/ngroups.+0.000001)
    # second, determine split feature from a on split node
    a = zeros(p, Tb)
    a_sum = sum(map(p->p.a, trees)) # has the same dim with a
    a_idx = argmax(a_sum, dims = 1)[findall(x->x==1.0, d)] # 1*Tb
    a[a_idx] .= 1
    # third, determine split value from b on groups with corresponding feature
    b = zeros(Tb)
    for i in 1:Tb
        # a_sum here is to prevent special case when d =1 and a[:,t] are all zeros
        if maximum(a[:,i])*maximum(a_sum[:,i]) != 0 
            fea = findall(x->round(x)==1.0, a[:,i])[1]
            slt_trees = trees[map(p->round(p.a[fea,i])==1, trees)]
            b[i] = mean(map(p->p.b[i], slt_trees))
        end
    end
    # forth, determine the active leaf node and label
    c = zeros(k, T)
    z = spzeros(n,Tl)
    for i in 1:n
        t = 1
        while t in 1:Tb
            if a[:,t]'*X[:,i] + 1e-12 >= b[t]
                t = 2*t+1
            else
                t = 2*t
            end
        end
        z[i,t-Tb] = 1
        c[:,t] = c[:,t]+y[:,i] # get node accumulate counts
    end
    # set c to be binary indicator of label
    crr = 0
    for i in Tl:T
        if maximum(c[:,i]) != 0
            c_idx = argmax(c[:,i])
            crr = crr+c[c_idx,i] # get the correct classified points in node i
            c[:,i] = zeros(k)
            c[c_idx,i] = 1
        end
    end
    return Tree(a,b,c,d,z,D), n-crr # return final tree and number of misclassified points
end


# prediction function
function predict_oct(X, y, a, b, c)
    ~,n = size(X)
    Tb_idx = length(b)

    if c isa JuMP.Containers.DenseAxisArray
        if c.data isa Matrix
            Tl_idx = size(c)[2]
        else
            Tl_idx = length(c)
        end
    else # c is vector or matrix
        if c isa Vector
            Tl_idx = length(c)-Tb_idx
        else # c isa matrix
            Tl_idx = size(c)[2]-Tb_idx
        end 
    end

    T = Tb_idx + Tl_idx
    
    pred = Array{Float64}(undef, n)
    z_pos = Array{Float64}(undef, n)
    for i in 1:n
        t = 1
        while !(t in (Tb_idx+1):T)
            if a[:,t]'*X[:,i] + 1e-12 >= b[t]
                t = 2*t+1
            else
                t = 2*t
            end
        end
        pred[i] = leaf_label(c,t)
        z_pos[i] = t-Tb_idx
    end
    crr = filter(x -> x == 0, pred-y)
    accr = length(crr)/n
    return accr, pred, z_pos
end

function objv_cost(X, Y, n, tree, alpha, L_hat)
    if X === nothing || Y === nothing
        return Inf, [-1]
    end
    Tb = length(tree.d)
    objv = 0
    z_pos = Array{Int64}(undef, n)
    for i in 1:n::Int64
        t = 1
        while t in 1:Tb
            if tree.a[:,t]'*X[:,i] + 1e-12 >= tree.b[t]
                t = 2*t+1
            else
                t = 2*t
            end
        end
        objv += Y[:,i] == tree.c[:,t] ? 0 : 1
        z_pos[i] = t-Tb
    end
    return 1/L_hat*objv + alpha*sum(tree.d), z_pos
end



# genrate a tree search root solution with given bound
function coor_descent_cart(X, Y, tree_w, lower, upper, alpha, L_hat)
    p,n = size(X)
    Tb = length(tree_w.d)
    tree = Trees.copy_tree(tree_w)
    for i in 1:Tb
        for j in 1:p
            tree.a[:,i] = zeros(p)
            tree.a[j,i] = 1
        end
    end
end


# end of the module
end

