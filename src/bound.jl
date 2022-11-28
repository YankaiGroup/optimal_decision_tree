module bound

using SparseArrays
using Trees

#according to a, b can be bounded more
#function init_bound(X, K, D, lower=nothing, upper=nothing)
function init_bound(p, n, K, D, lower=nothing, upper=nothing, val::Int64=0)
    Tb = 2^D-1
    Tl = 2^D
    lower_b = zeros(Tb) # repeat([minimum(X)], Tb)
    upper_b = ones(Tb) # repeat([maximum(X)], Tb)

    if lower !== nothing
        lower_b = min.(upper.b .- 1e-4, max.(lower.b, lower_b)) 
        upper_b = max.(lower.b + 1e-4, min.(upper.b, upper_b))
    end

    if val == 0 # if val == 0, it is not in validation mode
        lower_d = zeros(Tb) # [1.0,0.0,1.0] # 
        upper_d = ones(Tb) # [1.0,0.0,1.0] # 
    else # d will be a vector converted from integer
        d = float(reverse(digits(val, base=2, pad=Tb)))
        lower_d = copy(d)
        upper_d = copy(d)
    end
    lower = Tree(zeros(p, Tb), lower_b, zeros(K, Tb+Tl), lower_d, spzeros(n, Tl), D) # z init to all zero
    upper = Tree(ones(p, Tb), upper_b, hcat(zeros(K,Tb), ones(K,Tl)), upper_d, sparse(ones(n,Tl)), D) # z should be all one but too large, could setup after CF
    update_bound!(lower, upper)
    return lower, upper
end


function bound_idx(lowerx, upperx)
    x = (lowerx .!= upperx) # if not equal, x=1,not determined, if equal, x=0, means determined
    if x isa BitVector
        udtm = Int64[]
        dtm = Int64[]
        i = 1
        for el in x
            @inbounds el == 0 ? push!(dtm, i) : push!(udtm, i)
            i += 1
        end
    else#if x isa BitMatrix # for matrix i is in CartesianIndex form
        udtm = CartesianIndex[]
        dtm = CartesianIndex[]
        m,n = size(x)
        for j in 1:n # column first and then row
            for i in 1:m
                @inbounds x[i,j] == 0 ? push!(dtm, CartesianIndex(i,j)) : push!(udtm, CartesianIndex(i,j))
            end
        end
    end
    return udtm, dtm
end

function bound_idx_z(lowerz, upperz)
    n, Tl = size(lowerz)
    # transpose for speed
    lz = lowerz'
    uz = upperz'
    z_udt = Vector{Vector{Int}}(undef, n)
    for i in 1:n::Int
        zs_udt = findall(x->x==1, lz[:,i] .!= uz[:,i])
        z_udt[i] = copy(zs_udt)
    end
    return z_udt
end

function warm_start_z(X, a, b, D)
    if length(X) == 0 # in case of empty X when node id larger than ngroups_all
        return spzeros(0,0)
    end

    n = size(X)[2]
    T = 2^(D+1)-1
    Tl = 2^D
    #z = zeros(n, T)
    J = Array{Int64}(undef, n)
    for i in 1:n::Int64
        t = 1
        while !(t in Tl:T)
            if a[:,t]'*X[:,i] + 1e-12 >= b[t]
                t = 2*t+1
            else
                t = 2*t
            end
        end
        J[i] = t-Tl+1 # t is in Tl:T, but z is in 1:Tl
        #z[i,t] = 1
    end 
    return sparse(1:n,J,ones(n), n, Tl)
end


function bound_b(lower, upper, sortX, n, D)
    fathom = false
    for t in 1:(2^D-1)::Int64
        has_sol = false # initially false for each split node
        if upper.d[t] == 1 && lower.d[t] == 1 # only when d[t]=1, then need to check the feasibility
            fea_list = findall(x->x==1, upper.a[:,t])
            for fea in fea_list
                fea_v = sortX[fea,:]
                l_idx = searchsortedlast(fea_v, lower.b[t]) # >= lb's last pos
                if l_idx < 1 # this feature lb is less than min(fea_v)
                    l_idx += 1
                end
                u_idx = searchsortedfirst(fea_v, upper.b[t]) # <= ub's first pos
                if u_idx > n # this feature lb is larger than max(fea_v)
                    u_idx -= 1
                end
                # check if l and u already the smallest and the largest value of this feature
                if l_idx < n && lower.b[t] > fea_v[l_idx] # length(fea_v) == n
                    l_idx += 1 # >= lb's last pos +1
                end
                if u_idx > 1 && upper.b[t] < fea_v[u_idx]
                    u_idx -= 1 # <= ub's first pos -1
                end
                l_bt = sortX[fea,l_idx] # bound to larger one
                u_bt = sortX[fea,u_idx] # bound to smaller one

                if l_bt <= u_bt # if smaller or equal mean there exist a solution # can be equal but cannot be large
                    has_sol = true
                    break
                end
            end
        else
            has_sol = true
        end
        if !has_sol
            fathom = true
            break
        end
    end # the lower and upper is not changed, thus has to use fathom to check the feasibility
    return fathom
end

function check_bound_b(lower, upper, sortX::Matrix)
    p, n = size(sortX)
    D = lower.D
    #a_udt, a_dt = bound_idx(lower.a, upper.a)
    fathom = bound_b(lower, upper, sortX, n, D)
    return fathom
end

function boundIdx_all(lower::Tree, upper::Tree, z_udt::Vector{Vector{Int}})::Array
    a_udt, a_dt = bound_idx(lower.a, upper.a)
    d_udt, d_dt = bound_idx(lower.d, upper.d)
    c_udt, c_dt = bound_idx(lower.c, upper.c)
    # only when all a are determined, b will doing bound tightening on feature value
    if length(z_udt) == 0
        z_udt = bound_idx_z(lower.z, upper.z)
    end
    dtm_idx = [a_udt, a_dt, d_udt, d_dt, c_udt, c_dt, z_udt]
    return dtm_idx
end



# bound updating functions
function getchild!(list_b, list_l, idx, Tb)
    if idx > Tb
        push!(list_l, idx)
    else
        push!(list_b, idx)
        getchild!(list_b, list_l, idx*2, Tb)
        getchild!(list_b, list_l, idx*2+1, Tb)
    end
end

function getparent!(list, idx)
    while idx > 0
        pushfirst!(list, idx)
        idx = Int(floor(idx/2))
    end
end

function update_a!(bound, bVarIdx, direct)
    if direct == "left"
        bound.a[bVarIdx] = 0
    else # direct == right
        col = bVarIdx[2]
        bound.a[:,col] .= 0 # set all other on this Tb to zero
        bound.a[bVarIdx] = 1 # except for bVarIdx
    end
end

function update_b!(bound, bVarIdx, bValue)
    bound.b[bVarIdx] = bValue
end

function update_c!(bound, bVarIdx, direct)
    if direct == "left"
        bound.c[bVarIdx] = 0
    else # direct == right
        col = bVarIdx[2]
        bound.c[:,col] .= 0 # set all other on this Tb to zero
        bound.c[bVarIdx] = 1 # except for bVarIdx
    end
end

function update_d!(bound, bVarIdx, direct)
    if direct == "left"
        list_b = []
        list_l = []
        # descendants of bVarIdx and itself are all set to zero
        getchild!(list_b, list_l, bVarIdx, 2^bound.D-1)
        # for both lower and upper
        bound.d[list_b] .= 0
        bound.a[:,list_b] .= 0
        bound.b[list_b] .= 0
        pop!(list_l) # the last leaf (right most) is not determined
        bound.c[:,list_l] .= 0 
        # all list_l are not considered for z (this is also done in CF)
    else # direct == right
        list = []
        getparent!(list, bVarIdx)
        bound.d[list] .= 1 # all its parent node are set to 1
    end
end

function update_zs!(lower, upper, s, z_udt)
    Tl = 2^lower.D
    z_dt = setdiff(1:Tl, z_udt)
    upper.z[s, z_dt] .= 0 # here determined z are all set to zero
    # no lower update so that everytime there's one leaf not determined, 
    # however, in opt_func, if length of z_udt==1, z[i,z_udt[1]] will assign to 1 automatically
end

function update_bound!(lower, upper, bVar, bVarIdx, bValue=nothing, direct = "left")
    if bVar == "b"
        if direct == "left"
            update_b!(upper, bVarIdx, bValue)
        else # direct == "right"
            update_b!(lower, bVarIdx, bValue)
        end
    elseif bVar == "a"
        if direct == "left"
            update_a!(upper, bVarIdx, direct)
            col = bVarIdx[2]
            if sum(upper.a[:,col]) == 1 # check if there only one non-determined
                lower.a[:,col] .= copy(upper.a[:,col]) # then this col can all be determined
            end
        else # direct == "right"
            update_a!(upper, bVarIdx, direct)
            update_a!(lower, bVarIdx, direct)
        end
    elseif bVar == "c"
        if direct == "left"
            update_c!(upper, bVarIdx, direct)
            col = bVarIdx[2]
            if sum(upper.c[:,col]) == 1 # check if there only one non-determined
                lower.c[:,col] .= copy(upper.c[:,col])
            end
        else # direct == "right"
            update_c!(upper, bVarIdx, direct)
            update_c!(lower, bVarIdx, direct)
        end
    elseif bVar == "d"
        if direct == "left"
            update_d!(upper, bVarIdx, direct)
            update_d!(lower, bVarIdx, direct)
        else # direct == "right"
            update_d!(upper, bVarIdx, direct)
            update_d!(lower, bVarIdx, direct)
        end
    else
        println("wrong branching variable")
    end
end

function update_bound!(lower, upper)
    t = 1
    Tl = 2^lower.D
    nodelist = Int64[]
    push!(nodelist, t)
    while nodelist != []
        t = popfirst!(nodelist)
        if t >= Tl
            l_idx = findall(x->x==1, lower.c[:,t])
            u_idx = findall(x->x==1, upper.c[:,t])
            if length(l_idx) == 1
                upper.c[:,t] .= 0 # set all upper to zero
                upper.c[l_idx[1], t] = 1 # except l_idx
            elseif length(u_idx) == 1
                lower.c[:,t] .= 0 # set all lower to zero
                lower.c[u_idx[1], t] = 1 # except u_idx
            else
                # nothing to change
            end
        else # t is split node 
            if lower.d[t] == 0 && upper.d[t] == 0
                list_b = Int64[]
                list_l = Int64[]
                # descendants of bVarIdx and itself are all set to zero
                getchild!(list_b, list_l, t, Tl-1)
                # for both lower and upper
                lower.d[list_b] .= 0
                lower.a[:,list_b] .= 0
                lower.b[list_b] .= 0
                upper.d[list_b] .= 0
                upper.a[:,list_b] .= 0
                upper.b[list_b] .= 0
                t = pop!(list_l) # the last leaf (right most) is not determined
                lower.c[:,list_l] .= 0 
                upper.c[:,list_l] .= 0 
                push!(nodelist, t)
            else 
                if lower.d[t] == 1 && upper.d[t] == 1
                    list = []
                    getparent!(list, t)
                    lower.d[list] .= 1 # all its parent node are set to 1
                    upper.d[list] .= 1
                end
                # check a
                l_idx = findall(x->x==1, lower.a[:,t])
                u_idx = findall(x->x==1, upper.a[:,t])
                if length(l_idx) == 1
                    upper.a[:,t] .= 0 # set all upper to zero
                    upper.a[l_idx[1], t] = 1 # except l_idx
                elseif length(u_idx) == 1
                    lower.a[:,t] .= 0 # set all lower to zero
                    lower.a[u_idx[1], t] = 1 # except u_idx
                else
                    # nothing to change
                end
                push!(nodelist, 2*t)
                push!(nodelist, 2*t+1)
            end
        end
    end
end

end