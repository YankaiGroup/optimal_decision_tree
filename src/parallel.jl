module parallel
using MPI
using StaticArrays, SparseArrays, Serialization

function init()
    if !MPI.Initialized()
        MPI.Init()
    end
end

function finalize()
    if MPI.Initialized()
        MPI.Finalize()
    end
end

function create_world()
    if MPI.Initialized()
        global comm = MPI.COMM_WORLD
    else
        global comm = nothing
    end
end

#=
# create a new communicator with a subset of processes
# if the data group is larger than the total number of processes, then use the old communicator
# if the data group is smaller than the total number of processes, then create the new communicator with size of data group
function create_comm(ngroups::Int64)
    if MPI.Initialized()
        if ngroups >= MPI.Comm_size(MPI.COMM_WORLD)
            global comm = MPI.COMM_WORLD
        else
            world_group = MPI.Comm_group(MPI.COMM_WORLD)
            sub_group = MPI.Group_incl(world_group, Base.collect(Cint, 1:ngroups))
            global comm = MPI.Comm_create_group(MPI.COMM_WORLD, sub_group, 0)
            println(MPI.COMM_WORLD)
        end
    else
        global comm = nothing
    end
end
=#

function get_comm()
    comm
end

function myid()
    if MPI.Initialized()
        if MPI.COMM_NULL != comm
            return MPI.Comm_rank(comm)
        else
            return -1
        end
    else
        return 0
    end
end

is_root() = (myid() == 0)

function root_println(x, rank::Int=0)
    if myid() == rank
        Base.println(x)
    end
end

function nprocs()
    if MPI.Initialized() && MPI.COMM_NULL != comm
        return MPI.Comm_size(comm)
    else
        return 1
    end
end

function barrier()
    if nprocs() > 1
        MPI.Barrier(comm)
    end
end

function getpartition()
    partitionlist
end

function get_max_list_length()
    mapreduce(x->length(x), max, mylist)
end

function partition_concat(scen)
    if MPI.COMM_NULL != comm
        global mylist = Array{Array{Int64,1},1}(undef, nprocs()) ### ？？？
        for i in 1:nprocs()
            mylist[i] = Array{Int64,1}(undef, 0)
        end
        q, r = divrem(scen, nprocs())
        assign_size = [i <= r ? q + 1 : q for i = 1:nprocs()]
        j = 1
        for i in 1:scen
            if length(mylist[j]) < assign_size[j]
                push!(mylist[j], i)
            else # if mylist is full, the new idx should push to next mylist
                j += 1
                push!(mylist[j], i)
                
            end
        end
        global partitionlist = mylist[myid()+1]
    else
        global mylist = Array{Array{Int64,1},1}(undef, 1)
        mylist[1] = [1]
        global partitionlist = mylist[1]
    end
end

function partition_shuffle(scen)
    global mylist = Array{Array{Int64,1},1}(undef, nprocs())
    for i in 1:nprocs()
        mylist[i] = Array{Int64,1}(undef, 0)
    end
    for i in 1:scen
        push!(mylist[mod(i-1,nprocs())+1], i)
    end
    global partitionlist = mylist[myid()+1]
end

function sum(x::Number)
    if nprocs() == 1
        return x
    else
        return MPI.Allreduce(x, +, comm)
    end
end

function sumv!(x::Vector{T}) where {T<:Number}
    if nprocs() == 1
        return x
    else
        return MPI.Allreduce!(x, +, comm)
    end
end

function deserialize!(serialized, counts::Vector{Cint}, x::Vector{T}) where {T}
    @assert length(serialized) == Base.sum(counts)
    sind = 1
    eind = 0
    for i in 1:length(counts)
        eind += counts[i]
        xs = MPI.deserialize(serialized[sind:eind])
        #println(xs)
        for j in xs
            push!(x, j)
        end
        sind += counts[i]
    end
end

function allcollect(x::Vector{T}) where {T}
    if nprocs() > 1
        x_serialized = MPI.serialize(x)
        counts::Vector{Cint} = MPI.Allgather!([length(x_serialized)], UBuffer(similar([1], nprocs()), 1), comm)
        collect_serialized = MPI.Allgatherv!(x_serialized, VBuffer(similar(x_serialized, Base.sum(counts)), counts), comm)
        x = Vector{T}()
        deserialize!(collect_serialized, counts, x)
    end
    return x
end

function allcollect_bit(x::BitVector)
    if nprocs() > 1
        counts::Vector{Cint} = MPI.Allgather!([length(x)], UBuffer(similar([1], nprocs()), 1), comm)
        #println(counts)
        x = MPI.Allgatherv!(x, VBuffer(similar(x, Base.sum(counts)), counts), comm)
    end
    return x
end

function collect(x::Vector{T}) where {T}
    if nprocs() > 1
        x_serialized = MPI.serialize(x)
        counts::Vector{Cint} = MPI.Allgather!([length(x_serialized)], UBuffer(similar([1], nprocs()), 1), comm)
        if is_root()
            x_serialized = MPI.Gatherv!(x_serialized, VBuffer(similar(x_serialized, Base.sum(counts)), counts), 0, comm)
            x = Vector{T}()
            deserialize!(x_serialized, counts, x)
        else
            MPI.Gatherv!(x_serialized, nothing, 0, comm)
            x = nothing
        end
    end
    return x
end

function collect(x::Matrix{T}) where {T}
    if nprocs() > 1
        if size(x,1) > 0
            x = vec(reinterpret(SVector{size(x,1),eltype(x)}, x))
        else
            x = Vector{T}[]
        end
        x_serialized = MPI.serialize(x)
        counts::Vector{Cint} = MPI.Allgather!([length(x_serialized)], UBuffer(similar([1], nprocs()), 1), comm)
        if is_root()
            x_serialized = MPI.Gatherv!(x_serialized, VBuffer(similar(x_serialized, Base.sum(counts)), counts), 0, comm)
            x = Vector{Vector{T}}()
            deserialize!(x_serialized, counts, x)
            x = reduce(hcat, x)
        else
            MPI.Gatherv!(x_serialized, nothing, 0, comm)
            x = nothing
        end
    end
    return x
end

function spread(x::Union{Nothing, Vector{T}}, list_all::Array{Array{Int64,1},1} = mylist) where {T}
    if nprocs() > 1
        x_serialized = UInt8[]
        if is_root()
            counts = Int64[]
            for list in list_all
                x_srl_tmp = MPI.serialize(x[list])
                append!(x_serialized, x_srl_tmp)
                append!(counts, length(x_srl_tmp))
            end 
        end
        ct = 0
        if is_root()
            ct = MPI.Scatter!(UBuffer(counts,1), Ref{Int64}(), 0, comm)[]
        else
            ct = MPI.Scatter!(nothing, Ref{Int64}(), 0, comm)[]
        end
        if is_root()
            x_serialized = MPI.Scatterv!(VBuffer(x_serialized, counts), MPI.Buffer(similar(x_serialized, ct)), 0, comm)
        else
            x_serialized = MPI.Scatterv!(nothing, MPI.Buffer(similar(x_serialized, ct)), 0, comm)
        end

        x = MPI.deserialize(x_serialized)
    end
    return x
end


function combine_dict(x::Dict{Int,Float64})
    if nprocs() > 1
        ks = Vector{Int}()
        vs = Vector{Float64}()
        for (k,v) in x
            push!(ks,k)
            push!(vs,v)
        end
        counts::Vector{Cint} = MPI.Allgather!([length(ks)], UBuffer(similar([1], nprocs()), 1), comm)
        if is_root()
            ks_collected = MPI.Gatherv!(ks, VBuffer(similar(ks, Base.sum(counts)), counts), 0, comm)
            vs_collected = MPI.Gatherv!(vs, VBuffer(similar(vs, Base.sum(counts)), counts), 0, comm)
            for i in eachindex(ks_collected) #1:length(ks_collected)
                x[ks_collected[i]] = vs_collected[i]
            end
        else
            MPI.Gatherv!(ks, nothing, 0, comm)
            MPI.Gatherv!(vs, nothing, 0, comm)
        end
    end
    return x
end

function combine_dict(x::Dict{Int,SparseVector{Float64}})
    if nprocs() > 1
        ks = Vector{Int}()
        vs = Vector{SparseVector{Float64}}()
        for (k,v) in x
            push!(ks,k)
            push!(vs,v)
        end
        counts::Vector{Cint} = MPI.Allgather!([length(ks)], UBuffer(similar([1], nprocs()), 1), comm)
        vs_collected = collect(vs)
        if is_root()
            @assert length(vs_collected) == Base.sum(counts)
            ks_collected = MPI.Gatherv!(ks, VBuffer(similar(ks, Base.sum(counts)), counts), 0, comm)
            for i in eachindex(ks_collected) #1:length(ks_collected)
                x[ks_collected[i]] = vs_collected[i]
            end
        else
            MPI.Gatherv!(ks, nothing, 0, comm)
        end
    end
    return x
end

function bcast(buf, rank::Cint=Int32(0))
    if nprocs() > 1
        buf = MPI.bcast(buf, rank, comm)
    end
    return buf
end

end