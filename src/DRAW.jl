module DRAW


using Knet, Sloth
import Knet: minibatch, params, train!
using Statistics, Random, Dates
import Base: push!, empty!

using ArgParse
using Images, ImageMagick
using JLD2
using MosaicViews


include(Knet.dir("data","mnist.jl"))
_etype = Float32
_atype = gpu() >= 0 ? KnetArray{_etype} : Array{_etype}
SEED = -1 # FIXME: find a better way


include("utils.jl")
include("layers.jl")
include("network.jl")
include("visualize.jl")
include("train.jl")


end # module
