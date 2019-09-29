using NPZ

include("modules.jl")

function myread(prefix, suffix, arrname, atype=_atype)
    arr = npzread(prefix*arrname*suffix)' .+ 0
    return convert(atype, arr)
end

function myread(prefix, suffix, arrname, T::Int)
    mylist = Array{Any}(undef, T)
    for t = 0:T-1
        mylist[t+1] = myread(prefix, suffix, arrname*"-"*string(t))
    end
    return mylist
end

PREFIX = "/scratch/users/ikesen16/draw-pytorch/"
SUFFIX = ".npy"
# gnoise = myread(PREFIX, SUFFIX, "noise")
# Z = myread(PREFIX, SUFFIX, "z")

# i, f, c, o
function torch2knet(wih, bih, whh, bhh)
    hidden = div(size(wih, 2), 4)

    _helper0(w, i) = w[:, (i-1)*hidden+1:i*hidden]
    _helper1(w, b, i) = (_helper0(w, i), _helper0(b, i))

    wxi, bxi = _helper1(wih, bih, 1)
    wxf, bxf = _helper1(wih, bih, 2)
    wxu, bxu = _helper1(wih, bih, 3)
    wxo, bxo = _helper1(wih, bih, 4)

    whi, bhi = _helper1(whh, bhh, 1)
    whf, bhf = _helper1(whh, bhh, 2)
    whu, bhu = _helper1(whh, bhh, 3)
    who, bho = _helper1(whh, bhh, 4)

    _helper3(w) = reshape(w, 1, length(w))
    knet_wb = mapreduce(
        _helper3,
        hcat,
        [wxi, wxf, wxu, wxo,
         whi, whf, whu, who,
         bxi, bxf, bxu, bxo,
         bhi, bhf, bhu, bho])
    return reshape(knet_wb, 1, 1, length(knet_wb))
end

# read for loss function steps
T = 10
# mus = myread(PREFIX, SUFFIX, "mu", T)
# sigmas = myread(PREFIX, SUFFIX, "sigma", T)
# logsigmas = myread(PREFIX, SUFFIX, "logsigmas", T)
# kl_terms = myread(PREFIX, SUFFIX, "kl_terms", T)
# x = myread(PREFIX, SUFFIX, "x")
# x_recons = myread(PREFIX, SUFFIX, "x_recons")

PREFIX = "/scratch/users/ikesen16/draw-pytorch/"

wih = myread(PREFIX, SUFFIX, "encoder.weight_ih")
whh = myread(PREFIX, SUFFIX, "encoder.weight_hh")
bih = myread(PREFIX, SUFFIX, "encoder.bias_ih")
bhh = myread(PREFIX, SUFFIX, "encoder.bias_hh")
encoder_w = torch2knet(wih, bih, whh, bhh)

wih = myread(PREFIX, SUFFIX, "decoder.weight_ih")
whh = myread(PREFIX, SUFFIX, "decoder.weight_hh")
bih = myread(PREFIX, SUFFIX, "decoder.bias_ih")
bhh = myread(PREFIX, SUFFIX, "decoder.bias_hh")
decoder_w = torch2knet(wih, bih, whh, bhh)

mu_w = myread(PREFIX, SUFFIX, "mu_linear.weight")' .+ 0
mu_b = myread(PREFIX, SUFFIX, "mu_linear.bias")' .+ 0

sigma_w = myread(PREFIX, SUFFIX, "sigma_linear.weight")' .+ 0
sigma_b = myread(PREFIX, SUFFIX, "sigma_linear.bias")' .+ 0

write_w = myread(PREFIX, SUFFIX, "dec_w_linear.weight")' .+ 0
write_b = myread(PREFIX, SUFFIX, "dec_w_linear.bias")' .+ 0


o = parse_options("--batchsize 64 --seed 1")
model = DRAW(
    o[:A], o[:B], o[:N], o[:T], o[:encoder_dim],
    o[:decoder_dim], o[:zdim])


model.encoder.w = Param(_atype(encoder_w))
model.decoder.w = Param(_atype(decoder_w))
model.qnetwork.mu_layer.w = Param(_atype(mu_w))
model.qnetwork.mu_layer.b = Param(_atype(mu_b))
model.qnetwork.logsigma_layer.w = Param(_atype(sigma_w))
model.qnetwork.logsigma_layer.b = Param(_atype(sigma_b))
model.write_layer.w = Param(_atype(write_w))
model.write_layer.b = Param(_atype(write_b))

init_opt!(model, o[:optim])

function get_grads()
    loss_values = []
    J = @diff loss(model, x; loss_values=loss_values)
    grads = []
    for ps in params(model)
       push!(grads, grad(J, ps))
    end
    return loss_values, grads
end

function mytrain!(model, grads)
    for (i,p) in enumerate(params(model))
        update!(value(p), grads[i], p.opt)
    end
end
