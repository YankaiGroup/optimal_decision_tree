module Trees

using Random, SparseArrays
using Luxor
using JuMP
using parallel
export Tree, Tree_cf, Tree_lmda, Tree_lmda_rnd, tree_plot, leaf_label, printTree

struct Tree
    a::Union{Nothing, Matrix{Float64}}
    b::Union{Nothing, Vector{Float64}}
    c::Union{Nothing, Matrix{Float64}}
    d::Union{Nothing, Vector{Float64}}
    z::Union{Nothing, SparseMatrixCSC{Float64, Int64}, Matrix{Float64}, Vector{Vector{Int}}}
    D::Int
end

# all z are set to n*Tl
Tree() = Tree(nothing, nothing, nothing, nothing, nothing, 0)
# Initialize Tree parameters, here c is in length of T= 2^(D+1)-1
Tree(p::Int,D::Int,n::Int,k::Int=1) = Tree(zeros(p, 2^D-1), rand(2^D-1), zeros(k, 2^(D+1)-1), zeros(2^D-1), spzeros(n, 2^D), D)
# Initialize Tree for closed form lower bound
# To save the memory, here c is k*Tl,
# z is only one vector with the first element indicate the sample
# d is set to all ones in default
Tree_cf(p::Int,D::Int,i::Int,k::Int=1) = Tree(zeros(p, 2^D-1), zeros(2^D-1), zeros(k, 2^D), ones(2^D-1), vcat(i, zeros(2^D)), D)

function equal_tree(t1::Tree, t2::Tree)::Bool
    return sum(t1.a .!= t2.a) == 0 && sum(t1.c .!= t2.c) == 0 && sum(t1.d .!= t2.d) == 0 && sum(abs.(t1.b .- t2.b)) <= 1e-6
end

function copy_tree(t::Tree)::Tree
    a = t.a === nothing ? nothing : copy(t.a)
    b = t.b === nothing ? nothing : copy(t.b)
    c = t.c === nothing ? nothing : copy(t.c)
    d = t.d === nothing ? nothing : copy(t.d)
    z = t.z === nothing ? nothing : copy(t.z)
    D = t.D === nothing ? nothing : copy(t.D)
    return Tree(a, b, c, d, z, D)
end

function printTree(tree::Tree)
    if parallel.is_root()
        show(stdout, "text/plain", tree.a)
        println()
        show(stdout, "text/plain", tree.b)
        println()
        show(stdout, "text/plain", tree.c)
        println()
        show(stdout, "text/plain", tree.d)
        println()
    end
end

function leaf_label(c,t)
    if c isa JuMP.Containers.DenseAxisArray
        c = c.data
        if c isa Vector
            t = t-length(c)+1
        else # c isa Matrix
            t = t-size(c)[2]+1
        end
    end

    if c isa Matrix
        idx = findall(x->x==1.0, c[:,t])
        if length(idx) == 0
            println("leaf $t has no label")
            return 0
        else
            return idx[1]
        end
    else # c is a vector
        return c[t]
    end
end


function getleafassign(c, z, D)
    Tb = 2^D-1
    T = 2^(D+1)-1
    clst_label = Int64[]
    clst_idx = Vector{Int64}[]
    #=if z isa JuMP.Containers.DenseAxisArray
        z = z.data
    else
        z = z[:,(Tb+1):T]
    end=#
    for t in (Tb+1):T
        idx = findall(x->x==1, z[:,t-Tb])
        if length(idx) >= 1
            push!(clst_label, leaf_label(c,t))
            push!(clst_idx, idx)
        end
    end
    return clst_label, clst_idx
end


# transfer z notion of leaf index to binary matrix 
function z2mat(zvec::Vector{Vector{Int}}, Tl::Int)::Matrix{Int}
    n = length(zvec)
    zmat = zeros(n, Tl)
    for i in 1:n
        zmat[i,zvec[i][1]] = 1
    end
    return zmat
end

# transfer z notion of binary matrix to leaf index
function z2vec(zmat::Matrix{Int})::Vector{Vector{Int}}
    Tl, n = size(zmat)
    zvec = Vector{Vector{Int}}(undef, n) #[Vector{Int}(undef, 1) for i in 1:n]
    for i in 1:n
        zvec[i] = findall(x->x==1, zmat[i,:])
    end
    return zvec
end

# function to plot the optimal decision tree
function plotOCT(t, tree, l_idx, tbl, col, lvl)
    if t in l_idx
        rect_w = 50
        rect_h = 25
        #println(lvl, col)
        pos = tbl[lvl, col]
        rect(pos-(rect_w/2,rect_h/2), rect_w, rect_h, :stroke)
        println("block: $col for leaf $t")
        lbl = Int(round(leaf_label(tree.c,t)))
        num = Int(round(sum(tree.z[:,t-length(l_idx)+1])))
        cntnt = string("$lbl, $num")
        textcentered(cntnt, pos.x, pos.y)
        #return 0
    else
        elps_w = 70
        elps_h = 45
        pos = (tbl[lvl, col]+tbl[lvl, col+1])/2
        #println("$col, $pos")
        ellipse(pos, elps_w, elps_h, :stroke)
        #println(t)
        if tree.d[t] == 0
            fea = "ns"
        else
            fea = findall(x->x==1, tree.a[:,t])[1]
        end
        
        val = round(tree.b[t], digits=4)
        cntnt = string("f: $fea\nv: $val")
        textcentered(cntnt, pos.x, pos.y)
        if tree.d[t] == 0
            #arrow(pos+(0,elps_h/2), Point(0, -65)) # arrow to right
            # col update: 2^(D-1)
            if lvl == tree.D
                plotOCT(t*2+1, tree, l_idx, tbl, col+1, lvl+1)
            else    
                plotOCT(t*2+1, tree, l_idx, tbl, col+2^(tree.D-1-lvl), lvl+1)
            end
        else
            if lvl == tree.D
                #arrow(O, Point(0, -65)) # arrow to right
                plotOCT(t*2+1,tree, l_idx, tbl, col+1, lvl+1)
                #arrow(O, Point(0, -65)) # arrow to left    
                plotOCT(t*2, tree,l_idx, tbl, col, lvl+1)
            else
                #arrow(O, Point(0, -65)) # arrow to right
                plotOCT(t*2+1, tree,l_idx, tbl, col+2^(tree.D-1-lvl), lvl+1)
                #arrow(O, Point(0, -65)) # arrow to left    
                plotOCT(t*2, tree,l_idx, tbl, col-2^(tree.D-1-lvl), lvl+1)
            end
        end
    end
end

function tree_plot(tree, model = "dim", dataname = "iris")
    D = tree.D
    if tree.c isa Vector
        model= "CART"
    end
    @png begin
        #Drawing(1200,600)
        l_idx = 2^D:(2^(D+1)-1)
        tbl = Table(D+1, 2^D, 70, 50) # D+1 row, 2^D column
        col = Int(length(l_idx)/2)
        plotOCT(1, tree, l_idx, tbl, col, 1)
    end 70*2^D+10 50*(D+1)+10 "img/$dataname-$D-$model.png"
end


end