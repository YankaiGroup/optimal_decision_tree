using DataFrames, CSV
using Random, Distributions, StatsBase
using Plots
using MLDataUtils, Clustering
using Distributed, SharedArrays
# load functions for branch&bound and data preprocess from self-created module
@everywhere begin
    if !("src/" in LOAD_PATH)
        push!(LOAD_PATH, "src/")
    end    
end 
@everywhere begin
    if !("test/" in LOAD_PATH)
        push!(LOAD_PATH, "test/")
    end
end


using TimerOutputs: @timeit, get_timer

using Trees, bound, parallel, Nodes
using opt_func, ub_func, lb_func, bb_func, data_process


# arg1=: maximum depth of the tree
# arg2=: Lower bound method
# arg3=: random seed value
# arg4=: mode (parallel or serial)
# arg5=: dataset name

# arg6=: data package name or number of clusters for toy data generation
# arg7=: number of points in a cluster for toy data generation
# arg8=: dimension of the toy data

D = parse(Int, ARGS[1]) # 2 # 
LB_method = ARGS[2] # "CF+MILP+SG" #
seed = parse(Int, ARGS[3]) # 1 # 
scheme = ARGS[4] # sl # par #  
dataname = ARGS[5] # 
if dataname == "toy"
    nclst = 3 # parse(Int, ARGS[6]) 
    clst_n = 50 #parse(Int, ARGS[7]) number of samples in each cluster
    d = 2 # parse(Int, ARGS[8])
elseif dataname == "iris"
    datapackage = "datasets"
else
    datapackage = "nothing" # ARGS[6] # 
end

if scheme == "par"
    using MPI
    parallel.init()
end

parallel.create_world()
parallel.root_println("Running $(parallel.nprocs()) processes.")
parallel.root_println("Start training $dataname with seed $seed.")

#############################################################
################# Main Process Program Body #################
#############################################################

##################### read and generate data #####################
if parallel.is_root()
    if dataname == "toy"
        data, lbl, K = read_data(dataname, clst_n=clst_n, nclst=nclst, d=d)
    else
        data, lbl, K = read_data(dataname; datapackage=datapackage)
    end
    println(size(data))

    Random.seed!(seed)
    train, valid, test = stratifiedobs((view(data,:,:), lbl), (0.5, 0.25))
    tr_x, tr_y = train
    va_x, va_y = valid
    te_x, te_y = test
    parallel.root_println(size(tr_x))
    parallel.root_println(size(va_x))
    parallel.root_println(size(te_x))

    L_hat = maximum(counts(lbl))/length(lbl) # L_hat is the error of predicting all points with the major class    
    alp = 0.05 #alpha # parameter of complexity of the tree

    X = Matrix(hcat(tr_x, va_x)) # Matrix(tr_x) # Matrix(data) # 
    y = Vector(vcat(tr_y, va_y)) # Vector(tr_y) # lbl # 
    Y_g = opt_func.label_bin(y, K, "glb")
    Y = Y_g # setup global Y for label binarization
    Y_d = opt_func.label_bin(y, K, "dim")
    p,n = size(X)
    println("dimension: $p, class: $K")
    
    # Heuristic method for comparison and warm_start
    time_w = @elapsed tree_w, objv_w = CART_base(X, Y_g, K, D, alp, L_hat)
    println("cart cost: $objv_w")
    
else
    K = 0
    alp = 0
    L_hat = 0
    objv_w = 0
    tree_w = nothing
end

##################### Testing of different ODT methods #####################
# D and LB_method are already in all processes at the beginning.
K = parallel.bcast(K)
alp = parallel.bcast(alp)
L_hat = parallel.bcast(L_hat)
objv_w = parallel.bcast(objv_w)
tree_w = parallel.bcast(tree_w)

LB_method = split(LB_method, "+") # CF must in the method
##################### Start training and optimization #####################

# final trainging with the best setting
gap_sg = nothing
if parallel.is_root()
    println("size of X: $(size(X))")
    if "base-dim" in LB_method
        time_sg = @elapsed tree_sg, objv_sg, gap_sg, LB_sg = dimitris_OPT_cls(X, Y_d, K, D, alp, L_hat; warm_start = tree_w, mute=false, solver="CPLEX")
    elseif "base-glb" in LB_method
        time_sg = @elapsed tree_sg, objv_sg, gap_sg, LB_sg = global_OPT_DT(X, Y_g, K, D, alp, L_hat; warm_start = tree_w, mute=false, solver="CPLEX")
    else # LB_method == "CF+MILP+SG" or "CF+MILP"
        time_sg = @elapsed tree_sg, objv_sg, calc, LB_sg = branch_bound(X, Y_g, K, D, tree_w, objv_w, alp, L_hat, LB_method, true, false)
    end
else
    if "CF" in LB_method || "MILP" in LB_method || "SG" in LB_method
        # the whole data are saved in root and scattered to other processes, in other processes, data input are nothing
        tree_sg, objv_sg, calc, LB_sg = branch_bound(nothing, nothing, K, D, tree_w, objv_w, alp, L_hat, LB_method, true, false)
    end
end

if parallel.is_root()
    Trees.printTree(tree_w)
    Trees.printTree(tree_sg)
    if gap_sg === nothing
        gap_sg = round(calc[end][end], digits=4)
    end
    # result evaluation
    accr_trw, accr_trg, accr_w, accr_g = data_process.comp_result(X, y, te_x, te_y, tree_w, objv_w, tree_sg, objv_sg, gap_sg)
    # print result
    println("Dataname\t time\t objv\t lb\t gap\t train_w\t train_g\t test_w\t test_g")
    println("$dataname\t $(round(time_sg, digits=2))\t $objv_sg\t $(round(LB_sg, digits=3))\t $gap_sg\t $accr_trw\t $accr_trg\t $accr_w\t $accr_g")
    
    ##################### Tree structure plot #####################
    plt = false
    if plt
        tree_plot(tree_w, "CART", dataname)
        tree_plot(tree_sg, "sglb", dataname)
        tree_plot(tree_g, "glb", dataname)
        tree_plot(tree_d, "dim", dataname)
    end
end

parallel.finalize()


