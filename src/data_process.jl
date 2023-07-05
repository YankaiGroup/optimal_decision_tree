module data_process

using DataFrames, CSV
using CategoricalArrays
using Random, Distributions
using LinearAlgebra
using MLDataUtils, StatsBase
using ub_func

export read_data, datapreprocess, cluster_eval, plotResult, nestedEval, sig_gen


function read_data(dataname; datapackage=nothing, clst_n=nothing, nclst=nothing, d=nothing)
    Random.seed!(1) #120
    if dataname == "toy"
        data, label = data_generation(clst_n, nclst, d) # d = 2
    else
        # real world dataset testing
        if datapackage == "nothing"
            if Sys.iswindows()
                data, label = data_preprocess(dataname, nothing, joinpath(@__DIR__, "..\\data\\"), "NA") # read data in Windows
            else
                data, label = data_preprocess(dataname, nothing, joinpath(@__DIR__, "../data/"), "NA") # read data in Mac
            end
        else
            data, label = data_preprocess(dataname, datapackage) # read iris data from datasets package
            
        end
    end
    # scale all feature within 0-1
    label = vec(label)
    k = nlabel(label)
    # scale X to [0,1]
    rev_fea = []
    for p in 1:size(data)[1]
        p_min = minimum(data[p,:])
        p_max = maximum(data[p,:])
        if p_max!=p_min
            push!(rev_fea, p)
            fea = (data[p,:].-p_min) / (p_max-p_min)
            data[p,:] .= fea
        end
    end
    println(setdiff(rev_fea, 1:size(data)[1]))
    data = data[rev_fea, :]
    return data, label, k
end

# function for data pre-processing, here missingchar will be a single character
function data_preprocess(dataname, datapackage = "datasets", path=nothing, missingchar=nothing, header=false)
    # read data
    if path === nothing # read data from r-package datasets
        data = dataset(datapackage, dataname)
    elseif missingchar === nothing
        data = CSV.read(joinpath(path, dataname), DataFrame, header = header)
    else
        data = CSV.read(joinpath(path, dataname), DataFrame, header = header, missingstring = missingchar)
        data = dropmissing(data)
    end
    # process to separate features and label
    v = data[:,ncol(data)]
    if v isa CategoricalArray{String,1,UInt8}
        v =  (1:length(levels(v.pool)))[v.refs]
    else # change arbitary numerical labels to label from 1
        lbl = zeros(size(data)[1],1)
        for (idx,val) in enumerate(sort(unique(v)))
            lbl[findall(v -> v == val, v)] .= idx
        end
        v = Int.(lbl) 
    end
    # return data(deleting the first index column) in transpose for optimization process (only for some dataset)
    # return convert(Matrix, data[:,2:(ncol(data)-1)])', v # seeds
    return Matrix(Float64.(data[:,1:(ncol(data)-1)]))', v  # iris convert(Matrix, iris) will have error in niagara
end 

function data_generation(clst_n=50, k=3, d=2)
    # data generation
    Random.seed!(1) # 120
    nclst = k # number of clusters that a generated toy-data has
    data = Array{Float64}(undef, d, clst_n * nclst) # initial data array (clst_n*k)*2 
    label = Array{Float64}(undef, clst_n * nclst) # label is empty vector 1*(clst_n*k)

    mu = reshape(sample(1:30, nclst * d), nclst, d)
    for i = 1:nclst
            sig = round.(sig_gen(sample(1:10, d)))
            #println(sig)
            clst = rand(MvNormal(mu[i,:], sig), clst_n) # data is 2*clst_n
            data[:,((i - 1) * clst_n + 1):(i * clst_n)] = clst
            label[((i - 1) * clst_n + 1):(i * clst_n)] = repeat([i], clst_n)
    end
    return data, Int.(label)
end

function sig_gen(eigvals)
    n = length(eigvals)
    Q, ~ = qr(randn(n, n))
    D = Diagonal(eigvals) 
    return Q*D*Q'
end



function comp_result(X, y, te_x, te_y, tree_w, objv_w, tree_g, objv_g, gap_g)

    # final prediction on testing set
    accr_trw, ~, ~ = ub_func.predict_oct(X, y, tree_w.a, tree_w.b, round.(tree_w.c))
    accr_trg, ~, ~ = ub_func.predict_oct(X, y, tree_g.a, tree_g.b, round.(tree_g.c))
    accr_w, pred_w = ub_func.predict_oct(te_x, te_y, tree_w.a, tree_w.b, round.(tree_w.c))
    accr_g, pred_g = ub_func.predict_oct(te_x, te_y, tree_g.a, tree_g.b, round.(tree_g.c))

    println("comparison (CRT and Glb)on prediction: ", countmap(pred_w.==pred_g))
    return round(accr_trw*100, digits=3), round(accr_trg*100, digits=3), round(accr_w*100, digits=3), round(accr_g*100, digits=3)
end

## data split function train validate test, validate set to choose α, and then model is trained on combination of train and validate

## fit(testing) function to test the in-sample, out-of-sample result

end