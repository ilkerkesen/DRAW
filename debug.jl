include("modules.jl")
SEED = 1
using AutoGrad: gcheck
o = parse_options("--batchsize 64 --optim SGD(;lr=0.001) --T 10")
model = DRAW(
    o[:A], o[:B], o[:N], o[:T], o[:encoder_dim],
    o[:decoder_dim], o[:zdim])
init_opt!(model, o[:optim])
dtrn, dtst = mnistdata(xtype=Array{Float32})
x, y = first(dtrn)
x = reshape(_atype(x), 784, size(x,4))
# gcheck(loss, model, x)

# to train
# train!(model, x)

# forward calculation
# loss_values = []
# J = @diff loss(model, x; loss_values=loss_values)
# loss_values array stores two different loss values
