module opt_func

using Printf
using JuMP
using CPLEX
using Random, SparseArrays
using InteractiveUtils
using Trees, bound, parallel

mingap = 1e-2
time_lapse = 60*5 # 1 mins
time_lapse_glb = time_lapse*60*4 # 4 hours


export global_OPT_DT_SG, global_OPT_DT, dimitris_OPT_cls, mingap

############# auxilary functions #############
# only works if y is in category type data
# construct the maxtrix with row of k and column of points, for faster access of data
# transfer ordinal label to binary coded label
function label_bin(y, k, alg = "dim")
    n = length(y)
    if alg == "dim"
        Y = -ones(k,n)
    else
        Y = zeros(k,n)
    end
    for i in 1:n
        Y[y[i],i] = 1
    end
    return Y
end

# transfer binary coded label to ordinal label
function label_int(Y)
    ~,n = size(Y)
    y = Vector{Int64}(undef, n)
    for i in 1:n
        y[i] = findall(x->x==1, Y[:,i])[1]
    end
    return y
end

function node_direct(t)
    idx = t
    Ar = Int64[]
    Al = Int64[]
    while idx != 1
        if idx % 2 == 1 # can not aliqut
            idx = Int(floor(idx/2))
            push!(Ar, idx)
        else # idx % 2 == 0
            idx = Int(idx/2)
            push!(Al, idx)
        end
    end
    return Al, Ar
end

function mini_dist(v)
    #v = sort(v)
    n = length(v)
    dist = view(v,2:n)-view(v,1:(n-1))
    # handle the special case when all elements are 0
    if all(dist .== 0)
        return 1e-10
    end
    # find the min among non-zero elements
    filter!(x -> x != 0, dist)
    
    return minimum(dist)
end


# functions for variables setup
function var_a(m, l_a, a_udt, a_dt, a_w, p, Tb, rlx = false)
    #a_udt, a_dt= bound_idx(l_a, u_a)
    if length(a_dt) == 0
        if rlx # relax to continuous [0,1]
            @variable(m, 0<=a[j in 1:p, t in 1:Tb]<=1, start = a_w[j,t]);
        else # fix binary
            @variable(m, a[j in 1:p, t in 1:Tb], Bin, start = a_w[j,t]);
        end
    else
        # variable a formation
        if rlx # relax to continuous [0,1]
            @variable(m, 0<=a_var[i in 1:length(a_udt)]<=1) # create udtm decision var
        else # fix binary
            @variable(m, a_var[i in 1:length(a_udt)], Bin) # create udtm decision var
        end
        a = Matrix(undef, p, Tb)
        for i in eachindex(a_udt) #1:length(a_udt) # i is scalar
            set_start_value(a_var[i], a_w[a_udt[i]])
            a[a_udt[i]] = a_var[i]
        end
        for i in a_dt # i is CartesianIndex
            a[i] = l_a[i]
        end
    end
    return m, a
end

function var_d(m, l_d, d_udt, d_dt, d_w, Tb, rlx = false)
    #d_udt, d_dt= bound_idx(l_d, u_d)
    if length(d_dt) == 0
        if rlx
            @variable(m, 0<=d[t in 1:Tb]<=1, start = d_w[t]);
        else
            @variable(m, d[t in 1:Tb], Bin, start = d_w[t]);
        end
    else
        if rlx
            @variable(m, 0<=d_var[t in 1:length(d_udt)]<=1);
        else
            @variable(m, d_var[t in 1:length(d_udt)], Bin);
        end
        d = Vector(undef, Tb)
        for i in eachindex(d_udt) #i in 1:length(d_udt)
            set_start_value(d_var[i], d_w[d_udt[i]])
            d[d_udt[i]] = d_var[i]
        end 
        for i in d_dt # i is CartesianIndex
            d[i] = l_d[i]
        end
    end
    return m, d
end

function var_c(m, l_c, c_udt, c_dt, c_w, K, T, Tb, rlx = false)
    if length(c_dt) == 0
        if rlx
            @variable(m, 0<=c[k in 1:K, t in (Tb+1):T]<=1, start = c_w[k,t])
        else
            @variable(m, c[k in 1:K, t in (Tb+1):T], Bin, start = c_w[k,t])
        end
    else
        if rlx
            @variable(m, 0<=c_var[i in 1:length(c_udt)]<=1) # create udtm decision var
        else
            @variable(m, c_var[i in 1:length(c_udt)], Bin) # create udtm decision var
        end
        c = Matrix(undef, K, T) # the first Tb*k is useless, should reduce if memory issue raises
        for i in eachindex(c_udt) #1:length(c_udt) # i is scalar
            set_start_value(c_var[i], c_w[c_udt[i]])
            c[c_udt[i]] = c_var[i]
        end
        for i in c_dt # i is CartesianIndex
            c[i] = l_c[i]
        end
    end
    return m, c
end

# z are all in 1:Tl
################ z don't need to model if cost is determined
function var_z(m, l_z, z_udt, z_w, n, T, Tb, rlx = false)
    z_var_num = sum(filter(x->x>1, length.(z_udt)))
    if z_var_num == n*(Tb+1)
        if rlx
            @variable(m, 0<=z[i in 1:n, t in 1:(Tb+1)]<=1, start = z_w[i,t])
        else
            @variable(m, z[i in 1:n, t in 1:(Tb+1)], Bin, start = z_w[i,t])
        end
    else
        if rlx
            @variable(m, 0<=z_var[i in 1:z_var_num]<=1) # create udtm decision var
        else
            @variable(m, z_var[i in 1:z_var_num], Bin) # create udtm decision var
        end
        z = Matrix(undef, n, Tb+1) # set to 1:Tl
        j = 1
        for i in 1:n::Int
            z_i = z_udt[i] # here z_udt idxs are in 1:Tl
            if length(z_i) == 1 # if sample i only reaches one leaf then z[i] is all determined
                z[i,:] .= 0
                t = z_i[1]
                z[i,t] = 1 #l_z[i,t]
            else
                for t in 1:(Tb+1)
                    if t in z_i
                        set_start_value.(z_var[j], z_w[i,t])
                        z[i,t] = z_var[j]
                        j += 1
                    else
                        z[i,t] = l_z[i,t]
                    end
                end
            end
        end
    end
    return m, z
end

function var_z_sg(m, z_udt, z_w, n, T, Tb, rlx = false)
    z_var_num = length(z_udt)
    if z_var_num == Tb+1
        if rlx
            @variable(m, 0<=z[i in 1:n, t in (Tb+1):T]<=1, start = z_w[i,t])
        else
            @variable(m, z[i in 1:n, t in (Tb+1):T], Bin, start = z_w[i,t])
        end
    else
        if rlx
            @variable(m, 0<=z_var[i in 1:n, t in 1:length(z_udt)]<=1) # create udtm decision var
        else
            @variable(m, z_var[i in 1:n, t in 1:length(z_udt)], Bin) # create udtm decision var
        end
        z = Matrix(undef, n, T) # the first n*Tb is useless, should consider if memory issue raises
        j = 1
        for t in 1:T::Int
            if t in z_udt
                set_start_value.(z_var[:,j], z_w[:,t])
                z[:,t] = z_var[:,j]
                j += 1
            else
                z[:,t] .= zeros(n)
            end
        end
    end
    return m, z
end

# functions for constraints setup
# constraints for decomposible solver split node
#cost_udt on n, z_udt on Tl
function cons_setup_node(m, X, p, z_udt, Tb, eps, eps_max, a, b, d, z)
    @constraint(m, [t in 1:Tb], sum(a[j,t] for j in 1:p) == d[t])
    @constraint(m, [t in 1:Tb], b[t] <= d[t]);
    @constraint(m, [t in 2:Tb], d[t] <= d[Int(floor(t/2))]) # p(t) is floor(t/2) for t's parent node index
    # the following constraints are set for DT spliting track
    n_idx = findall(x->length(x)>1, z_udt)
    for i in n_idx
        for t in z_udt[i]
            Al, Ar = node_direct(t+Tb)
            #@constraint(m, [i in 1:n, mt in Al], sum(a[j,mt]*(X[j,i]+eps[j]-eps_min) for j in 1:p)+eps_min <= b[mt]+(1+eps_max)*(1-z[i,t-Tb])) # eq 13
            @constraint(m, [mt in Al], sum(a[j,mt]*(X[j,i]+eps[j]) for j in 1:p) + (1+eps_max)*(1-d[mt]) <= b[mt]+(1+eps_max)*(1-z[i,t])) # eq 13
            @constraint(m, [mt in Ar], sum(a[j,mt]*X[j,i] for j in 1:p) >= b[mt]-(1-z[i,t]))
        end
    end
    return m
end

# constraints for MILP solver leaf
function cons_setup_leaf_MILP(m, y, n, K, T, Tb, c, z_udt, costs)
    # leaf node constraints
    # constraints for binary node label variable ckt
    @constraint(m, [t in (Tb+1):T], sum(c[k,t] for k in 1:K) <= 1) # for each node, label should at most choose one label
    # constraints for determine which node the point i will be allocated
    @constraint(m, [i in 1:n], costs[i] >= 1-sum(y[k,i]*c[k,t+Tb] for k in 1:K for t in z_udt[i]))
    return m
end

# constraints for SG solver leaf
function cons_setup_leaf_SG(m, y, n, K, T, Tb, c, z, z_udt, costs)
    # leaf node constraints
    # constraints on z[i,t] at leaf
    n_idx = findall(x->length(x)>1, z_udt)
    @constraint(m, [i in n_idx], sum(z[i,t] for t in 1:(Tb+1)) == 1) # guarantee a sample only in one leaf
    # constraints for binary node label variable ckt
    @constraint(m, [t in (Tb+1):T], sum(c[k,t] for k in 1:K) <= 1) # for each node, label should at most choose one label
    # constraints for determine which node the point i will be allocated
    if z_udt isa Vector{Int64}# for only SG mode
        @constraint(m, [i in 1:n, t in (Tb+1):T], costs[i] >= z[i,t-Tb]-sum(y[k,i]*c[k,t] for k in 1:K))
    else
        @constraint(m, [i in 1:n, t in z_udt[i]], costs[i] >= z[i,t]-sum(y[k,i]*c[k,t+Tb] for k in 1:K))
    end        
    return m
end

# initialize the optimizer
function optimizer_init(mute, time_lapse)
    m = direct_model(CPLEX.Optimizer());
    if mute
        set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 0)
    end
    set_optimizer_attribute(m, "CPX_PARAM_THREADS",1)
    set_optimizer_attribute(m, "CPX_PARAM_TILIM", time_lapse) # maximum runtime limit is 1 hours
    # here the gap should always < mingap of BB, e.g. if mingap = 0.1%, then gap here should be < 0.1%, the default is 0.01%
    set_optimizer_attribute(m, "CPX_PARAM_EPGAP", mingap/2) 
    return m
end


##### if y is float, then using regression method else, y is integer or binary, then use classification method #####
function global_OPT_DT_SG(X, Y, K, D, alpha, L_hat; lower=nothing, upper=nothing, eps=nothing, dtm_idx=nothing, w_sos=nothing, lambda = nothing, warm_start = nothing, mute=false, rlx = false, time = time_lapse)
    # parameter setup
    p, n = size(X)
    T = 2^(D+1)-1
    Tb = Int(floor(T/2))
    eps_min = minimum(eps)
    eps_max = maximum(eps)

    a_udt = dtm_idx[1]
    a_dt = dtm_idx[2]
    d_udt = dtm_idx[3]
    d_dt = dtm_idx[4]
    c_udt = dtm_idx[5]
    c_dt = dtm_idx[6]
    z_udt = dtm_idx[7]
    # check warm_start status
    if warm_start === nothing
        warm_start = Tree(zeros(p, Tb), rand(Tb), zeros(K, T), zeros(Tb), spzeros(n, Tb+1), D)
    end

    # optimizer setup
    m = optimizer_init(mute, time)

    # variables setup
    # ajt: bool, choose which feature to split; real, mixed feature(hyperplane) to split
    m, a = var_a(m, lower.a, a_udt, a_dt, warm_start.a, p, Tb, rlx)
    # bt: real, choose which value to split
    @variable(m, lower.b[t] <= b[t in 1:Tb] <= upper.b[t], start = warm_start.b[t]); 
    # dt: bool, 1 if t has a split, 0 otherwise
    m, d = var_d(m, lower.d, d_udt, d_dt, warm_start.d, Tb, rlx)
    # variables for leaf
    # ckt: from class 1 to k
    m, c = var_c(m, lower.c, c_udt, c_dt, warm_start.c, K, T, Tb, false)
    # zit: 1 if sample i in node t, t include both branch and leaf nodes.
    if z_udt isa Vector{Int64}# for only SG mode
        m, z = var_z_sg(m, z_udt, warm_start.z, n, T, Tb, rlx) # used in only SG mode
    else
        m, z = var_z(m, lower.z, z_udt, warm_start.z, n, T, Tb, rlx) # used in CF+SG mode
    end
    # variable for the loss
    @variable(m, 0<=costs[i in 1:n]<=1) # loss of sample i among all node 
    # constraints setup
    m = cons_setup_node(m, X, p, z_udt, Tb, eps, eps_max, a, b, d, z)
    m = cons_setup_leaf_SG(m, Y, n, K, T, Tb, c, z, z_udt, costs)
    # objective
    @objective(m, Min, 1/L_hat*sum(costs[i] for i in 1:n) + alpha*sum(d[t] for t in 1:Tb));
    optimize!(m);
    a = value.(a)
    b = value.(b)
    c = round.(value.(c))
    d = value.(d) 
    objv = objective_bound(m)
    cpm = backend(m)
    prob_type = CPXgetprobtype(cpm.env, cpm.lp)
    if prob_type == CPXPROB_LP
        gap = 0.0
    else
        gap = relative_gap(m)
    end
    tree = Tree(a,b,c,d,nothing,D) # or tree = Tree(a,b,c,z_udt,D) so that z_udt is used in decesendent
    return tree, objv, gap 
end


function global_OPT_DT_MILP(X, Y, K, D, alpha, L_hat; lower=nothing, upper=nothing, dtm_idx=nothing, warm_start = nothing, mute=false)
    # parameter setup
    p, n = size(X)
    T = 2^(D+1)-1
    Tb = Int(floor(T/2))
    c_udt = dtm_idx[5]
    c_dt = dtm_idx[6]
    z_udt = dtm_idx[7]
    # check warm_start status
    if warm_start === nothing
        warm_start = Tree(zeros(p, Tb), rand(Tb), zeros(K, T), zeros(Tb), spzeros(n, T), D)
    end
    # optimizer setup
    m = optimizer_init(mute, time_lapse)
    # variables for leaf
    # ckt: from class 1 to k
    m, c = var_c(m, lower.c, c_udt, c_dt, warm_start.c, K, T, Tb, false)
    # variable for the loss
    @variable(m, 0<=costs[i in 1:n]<=1) # loss of sample i among all node 
    # constraints setup
    m = cons_setup_leaf_MILP(m, Y, n, K, T, Tb, c, z_udt, costs)
    # objective
    @objective(m, Min, 1/L_hat*sum(costs[i] for i in 1:n) + alpha*sum(lower.d));
    optimize!(m);
    # retrive value and cost
    c = round.(value.(c))
    objv = objective_bound(m)
    return c, objv
end


#############################################################
# The following two solvers are used for benchmark comparison
############# global optimization solvers #############
##### if y is float, then using regression method else, y is integer or binary, then use classification method #####
function global_OPT_DT(X, y, K, D, alpha, L_hat, w_sos=nothing; lambda = nothing, warm_start = nothing, mute=false, solver="CPLEX")
    # check the problem type
    if y isa Vector 
        type = "reg"
    else # y is a category matrix: K*n
        type = "cls"
    end

    p, n = size(X)
    T = 2^(D+1)-1
    Tb = Int(floor(T/2))
    sortX = sort(X, dims=2) # sorted on each feature
    eps = vec(mapslices(mini_dist, sortX, dims=2))
    eps_min = minimum(eps)
    eps_max = maximum(eps)

    # check warm_start status
    if warm_start !== nothing
        a_start = warm_start.a
        b_start = warm_start.b
        d_start = warm_start.d
        c_start = warm_start.c
        z_start = warm_start.z
    else
        a_start = zeros(p, Tb)
        b_start = rand(Tb)
        d_start = zeros(Tb)
        z_start = zeros(n, Tb+1)
        if type == "reg"
            c_start = rand(T)
        else
            c_start = zeros(K, T)
        end
    end

    #w_bin = rlt.centers # weight of the binary variables
    if solver=="CPLEX"
        m = Model(CPLEX.Optimizer);
        if mute
            set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 0)
        end
        set_optimizer_attribute(m, "CPX_PARAM_THREADS",1)
        set_optimizer_attribute(m, "CPX_PARAM_TILIM", time_lapse_glb) # maximum runtime limit is 4 hours
        # here the gap should always < mingap of BB, e.g. if mingap = 0.1%, then gap here should be < 0.1%, the default is 0.01%
        # set_optimizer_attribute(m, "CPX_PARAM_EPGAP", 0.05) 
    end
    
    @variable(m, a[j in 1:p, t in 1:Tb], Bin, start = a_start[j,t]); # ajt: bool, choose which feature to split; real, mixed feature(hyperplane) to split
    @variable(m, 0 <= b[t in 1:Tb] <= 1, start = b_start[t]); # bt: real, choose which value to split
    @variable(m, d[t in 1:Tb], Bin, start = d_start[t]); # dt: bool, 1 if t has a split, 0 otherwise

    @constraint(m, [t in 1:Tb], sum(a[j,t] for j in 1:p) == d[t])
    @constraint(m, [t in 1:Tb], b[t] <= d[t]);
    @constraint(m, [t in 2:Tb], d[t] <= d[Int(floor(t/2))]) # p(t) is floor(t/2) for t's parent node index

    @variable(m, z[i in 1:n, t in (Tb+1):T], Bin, start = z_start[i,t-Tb]); # zit: 1 if sample i in node t, t include both branch and leaf nodes.
    # the following constraints are set for DT spliting track
    for t in (Tb+1):T
        Al, Ar = node_direct(t)
        #@constraint(m, [i in 1:n, mt in Al], a[:,mt]'*X[:,i]+eps_min <= b[mt]+(1+eps_max)*(1-z[i,t])) # eq 13
        #@constraint(m, [i in 1:n, mt in Al], a[:,mt]'*(X[:,i]+eps.-eps_min)+eps_min <= b[mt]+(1+eps_max)*(1-z[i,t])) # eq 13
        @constraint(m, [i in 1:n, mt in Al], a[:,mt]'*(X[:,i]+eps) + (1+eps_max)*(1-d[mt]) <= b[mt]+(1+eps_max)*(1-z[i,t])) # eq 13
        @constraint(m, [i in 1:n, mt in Ar], a[:,mt]'*X[:,i] >= b[mt]-(1-z[i,t]))
    end

    # leaf node variables and constraints
    # constraints on z[i,t] at leaf
    @constraint(m, [i in 1:n], sum(z[i,t] for t in (Tb+1):T) == 1) # guarantee a sample only in one leaf
    # variables for leaf label and loss
    
    @variable(m, 0<=costs[i in 1:n]<=1) # loss of sample i among all node 
    if type == "reg"
        @variable(m, l[i in 1:n, t in (Tb+1):T], Bin) # loss for sample i at node t
        @variable(m, 0 <= c[t in (Tb+1):T], start = c_start[t]) # ct: from class 1 to k
        # constraints for binary node label variable ckt
        @constraint(m, [i in 1:n, t in (Tb+1):T], y[i]-c[t] <= l[i,t])
        # constraints for determine which node the point i will be allocated
        @constraint(m, [i in 1:n, t in (Tb+1):T], costs[i]-l[i,t] <= 2*(1-z[i,t]))
        @constraint(m, [i in 1:n, t in (Tb+1):T], costs[i]-l[i,t] >= -2*(1-z[i,t]))
        # objective
        @objective(m, Min, sum((1/L_hat*costs[i]^2+alpha/n*sum(d[t] for t in 1:Tb)) for i in 1:n));
    else # type == "cls"
        @variable(m, c[k in 1:K, t in (Tb+1):T], Bin, start = c_start[k,t]) # ckt: from class 1 to k
        # constraints for binary node label variable ckt
        @constraint(m, [t in (Tb+1):T], sum(c[k,t] for k in 1:K) <= 1) # for each node, label should at most choose one label
        # constraints for determine which node the point i will be allocated
        @constraint(m, [i in 1:n, t in (Tb+1):T], 0.5*sum(c[k,t]-2*y[k,i]*c[k,t] for k in 1:K)-costs[i] <= 0.5-z[i,t])
        #@constraint(m, [i in 1:n, t in (Tb+1):T, k in 1:K], y[k,i]+c[k,t]-2*y[k,i]*c[k,t]-costs[i] <= 1-z[i,t])
        #@constraint(m, [i in 1:n, t in (Tb+1):T], costs[i] >= z[i,t-Tb]-sum(y[k,i]*c[k,t] for k in 1:K))
        # objective
        @objective(m, Min, 1/L_hat*sum(costs[i] for i in 1:n)+alpha*sum(d[t] for t in 1:Tb));
    end

    optimize!(m);

    a = value.(a)
    b = value.(b)
    c = round.(value.(c))
    d = value.(d)
    z = value.(z)
    c = hcat(zeros(K,Tb),c.data)
    objv = objective_value(m)
    lb = objective_bound(m)
    gap = relative_gap(m)
    tree = Tree(a,b,c,d,sparse(z.data),D)
    return tree, objv, gap, lb
end


############# Dimitris OPT solvers ############# 
function dimitris_OPT_cls(X, Y, K, D, alpha, L_hat; warm_start = nothing, mute=false, solver="CPLEX", val=0, time=14400)
    p, n = size(X)
    T = 2^(D+1)-1
    Tb = Int(floor(T/2))
    sortX = sort(X, dims=2) # sorted on each feature
    eps = vec(mapslices(mini_dist, sortX, dims=2))
    eps_min = minimum(eps)
    eps_max = maximum(eps)

    N_min = 1 #ceil(0.05*n) # minimum number of points at each leaf

    # check warm_start status
    if warm_start !== nothing
        a_start = warm_start.a
        b_start = warm_start.b
        d_start = warm_start.d
        c_start = warm_start.c
        z_start = warm_start.z
    else
        a_start = zeros(p, Tb)
        b_start = rand(Tb)
        c_start = zeros(K, T)
        d_start = zeros(Tb)
        z_start = zeros(n, Tb+1)
    end

    # start modeling
    if solver=="CPLEX"
        m = Model(CPLEX.Optimizer);
        if mute
            set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 0)
        end
        #set_optimizer_attribute(m, "CPX_PARAM_THREADS",1)
        set_optimizer_attribute(m, "CPX_PARAM_TILIM", time) # maximum runtime limit is 1 hours # time_lapse_glb
        # here the gap should always < mingap of BB, e.g. if mingap = 0.1%, then gap here should be < 0.1%, the default is 0.01%
        # set_optimizer_attribute(m, "CPX_PARAM_EPGAP", 0.05) 
    end
    
    @variable(m, a[j in 1:p, t in 1:Tb], Bin, start = a_start[j,t]); # ajt: bool, choose which feature to split; real, mixed feature(hyperplane) to split
    @variable(m, 0 <= b[t in 1:Tb] <=1, start = b_start[t]); # bt: real, choose which value to split
    @variable(m, c[k in 1:K, t in (Tb+1):T], Bin, start = c_start[k,t]); # ckt = 1 if ct = k, ct = argmax_k{Nkt}, Nkt is the number of points of label k in node t
    # for hyperparameter tuning, change variable d to fix vector
    if val == 0
        @variable(m, d[t in 1:Tb], Bin, start = d_start[t]); # dt: bool, 1 if t has a split, 0 otherwise
    else
        d = float(reverse(digits(val, base=2, pad=Tb)))
    end

    @variable(m, z[i in 1:n, t in (Tb+1):T], Bin, start = z_start[i,t-Tb]); # zit: 1 if sample i in node t, t include both branch and leaf nodes.
    @variable(m, l[t in (Tb+1):T], Bin) # lt: 1 if leaf t contains any points
    @variable(m, 0 <= L[t in (Tb+1):T], start = rand())
    @variable(m, 0 <= Nkt[k in 1:K, t in (Tb+1):T], start=rand())
    @variable(m, 0 <= Nt[t in (Tb+1):T], start=rand())
    
    # eq 20
    @constraint(m, [k in 1:K, t in (Tb+1):T], L[t] >= Nt[t]-Nkt[k,t]-n*(1-c[k,t])) 
    # eq 21
    @constraint(m, [k in 1:K, t in (Tb+1):T], L[t] <= Nt[t]-Nkt[k,t]+n*c[k,t]) 
    # eq 15
    @constraint(m, [k in 1:K, t in (Tb+1):T], Nkt[k,t] == 1/2*sum((1+Y[k,i])*z[i,t] for i in 1:n)) 
    # eq 16
    @constraint(m, [t in (Tb+1):T], Nt[t] == sum(z[i,t] for i in 1:n))
    # eq 18 
    @constraint(m, [t in (Tb+1):T], sum(c[k,t] for k in 1:K) == l[t]) 

    # the following constraints are set for DT spliting track
    for t in (Tb+1):T
        Al, Ar = node_direct(t)
        # eq 13
        #@constraint(m, [i in 1:n, mt in Al], a[:,mt]'*X[:,i]+eps_min <= b[mt]+(1+eps_max)*(1-z[i,t])) # eq 13
        #@constraint(m, [i in 1:n, mt in Al], a[:,mt]'*(X[:,i]+eps.-eps_min)+eps_min <= b[mt]+(1+eps_max)*(1-z[i,t])) # eq 13
        @constraint(m, [i in 1:n, mt in Al], a[:,mt]'*(X[:,i]+eps) + (1+eps_max)*(1-d[mt]) <= b[mt]+(1+eps_max)*(1-z[i,t])) # eq 13
        # eq 14
        @constraint(m, [i in 1:n, mt in Ar], a[:,mt]'*X[:,i] >= b[mt]-(1-z[i,t])) 
    end
    # eq 8
    @constraint(m, [i in 1:n], sum(z[i,t] for t in (Tb+1):T) == 1) # guarantee a sample only in one leaf
    # eq 6
    @constraint(m, [i in 1:n, t in (Tb+1):T], z[i,t] <= l[t]) 
    # eq 7
    @constraint(m, [t in (Tb+1):T], sum(z[i,t] for i in 1:n) >= N_min*l[t]) 
    # eq 2
    @constraint(m, [t in 1:Tb], sum(a[j,t] for j in 1:p) == d[t])
    # eq 3
    @constraint(m, [t in 1:Tb], b[t] <= d[t]);
    # eq 5
    @constraint(m, [t in 2:Tb], d[t] <= d[Int(floor(t/2))]) # p(t) is floor(t/2) for t's parent node index
    #@constraint(m, [i in 1:n, t in (Tb+1):T], z[i,t] <= z[i, Int(floor(t/2))]) # guarantee sample in leaf must also exist in its parent node

    # objective
    @objective(m, Min, 1/L_hat*sum(L[t] for t in (Tb+1):T) + alpha*sum(d[t] for t in 1:Tb));
    optimize!(m);

    a = value.(a)
    b = value.(b)
    c = value.(c)
    d = value.(d)
    z = value.(z)
    c = hcat(zeros(K,Tb),c.data)
    objv = objective_value(m)
    lb = objective_bound(m)
    gap = relative_gap(m) 
    tree = Tree(a,b,c,d,sparse(z.data),D)
    return tree, objv, gap, lb
end





end
