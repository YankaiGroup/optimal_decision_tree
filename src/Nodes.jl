module Nodes

using Printf
using Trees, bound, parallel

export Node

struct Node
    lower::Union{Nothing, Tree}
    upper::Union{Nothing, Tree}
    level::Int
    LB::Float64
    costs::Union{Nothing, Vector{Float64}} # {-1, 0, 1}  -1: undetermined, 0, 1 are cost for sample i
    groups #::Union{Nothing, Vector{Vector{Int64}}}
    lambda
    group_trees::Union{Nothing, Tree, Vector{Tree}}
    LB_gp::Union{Nothing, Vector{Float64}}
    lrg_gap::Union{Nothing, Vector{Bool}}
    bch_var::String
end

Node() = Node(nothing, nothing, -1, -1e15, nothing, nothing, nothing, nothing, nothing, nothing, "d")


function node_init()
    node = Node()
    return node
end

# function to print the node in a neat form
function printNodeList(nodeList)
    for i in eachindex(nodeList) #1:length(nodeList)
        println(map(x -> @sprintf("%.3f",x), getfield(nodeList[i],:lower))) # reserve 3 decimal precision
        println(map(x -> @sprintf("%.3f",x), getfield(nodeList[i],:upper)))
        println(getfield(nodeList[i],:level)) # integer
        println(map(x -> @sprintf("%.3f",x), getfield(nodeList[i],:LB)))
    end
end


end