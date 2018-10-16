using Knet
import Knet: Knet, minibatch, params, train!

using Images
using ArgParse
using ImageMagick
using JLD2

using Statistics, Random, Dates
import Base: push!, empty!

include(Knet.dir("data","mnist.jl"))


_etype = gpu() >= 0 ? Float32 : Float64
_atype = gpu() >= 0 ? KnetArray{_etype} : Array{_etype}


function initwb(input_dim::Int, output_dim::Int, atype=_atype)
    w = param(output_dim, input_dim; init=xavier, atype=_atype)
    b = param(output_dim, 1; atype=_atype)
    return (w,b)
end


struct Linear
    w
    b
end


(l::Linear)(x) = l.w * x .+ l.b


function Linear(input_dim::Int, output_dim::Int, atype=_atype)
    w, b = initwb(input_dim, output_dim, atype)
    return Linear(w, b)
end


struct FullyConnected
    w
    b
    activate
end


(l::FullyConnected)(x) = activate.(l.w * x .+ l.b)


function FullyConnected(
    input_dim::Int, output_dim::Int, activate=relu, atype=_atype)
    w, b = initwb(input_dim, output_dim, atype)
    return FullyConnected(w, b, activate)
end


struct ReadNoAttention
end


(l::ReadNoAttention)(x, xhat, hdec) = vcat(x, xhat)


WriteNoAttention = Linear


struct QNet
    mu_layer
    logsigma_layer
end


function (l::QNet)(henc)
    mu = l.mu_layer(henc)
    logsigma = l.logsigma_layer(henc)
    sigma = exp.(logsigma)
    noise = randn!(similar(mu))
    sampled = mu .+ noise .* sigma
    return (sampled, mu, logsigma, sigma)
end


function QNet(input_dim::Int, output_dim::Int, atype=_atype)
    mu_layer = Linear(input_dim, output_dim, atype)
    logsigma_layer = Linear(input_dim, output_dim, atype)
    return QNet(mu_layer, logsigma_layer)
end


function sample_noise(q::QNet, batchsize::Int)
    zdim = size(value.(q.mu_layer.w), 2)
    z = randn(zdim, batchsize)
    atype = typeof(value.(q.mu_layer.w))
    return convert(atype, z)
end


mutable struct DRAWOutput
    mus
    logsigmas
    sigmas
    cs
end


DRAWOutput() = DRAWOutput([], [], [], [])


function push!(o::DRAWOutput, c)
    push!(o.cs, c)
end


function push!(o::DRAWOutput, mu, logsigma, sigma, c)
    push!(o.mus, mu)
    push!(o.logsigmas, logsigma)
    push!(o.sigmas, sigma)
    push!(o.cs, c)
end


function empty!(o::DRAWOutput)
    empty!(o.mus, mu)
    empty!(o.logsigmas, logsigma)
    empty!(o.sigma, sigma)
    empty!(o.cs, c)
end


mutable struct DRAW
    A
    B
    N
    T
    read_layer
    write_layer
    qnetwork
    encoder
    decoder
    encoder_hidden
    decoder_hidden
    state0
end


function DRAW(A, B, N, T, encoder_dim, decoder_dim, noise_dim, atype=_atype)
    encoder_dim = A*B
    read_layer = ReadNoAttention()
    write_layer = WriteNoAttention(decoder_dim, A*B, atype)
    qnetwork = QNet(encoder_dim, noise_dim, atype)
    encoder = RNN(2*N*N+decoder_dim, encoder_dim)
    decoder = RNN(noise_dim, decoder_dim)
    encoder_hidden = []
    decoder_hidden = []
    # dummy_state = atype(zeros(decoder.hiddenSize, 1))
    dummy_state = atype(zeros(decoder.hiddenSize, 1))

    return DRAW(
        A,
        B,
        N,
        T,
        read_layer,
        write_layer,
        qnetwork,
        encoder,
        decoder,
        encoder_hidden,
        decoder_hidden,
        dummy_state
    )
end


function DRAW(N, T, encoder_dim, decoder_dim, noise_dim, atype=_atype)
    A = B = N
    return DRAW(A, B, N, T, encoder_dim, decoder_dim, noise_dim, atype=_atype)
end


# reconstruct
function (model::DRAW)(x)
    empty!(model.encoder_hidden)
    empty!(model.decoder_hidden)
    output = DRAWOutput()
    atype = typeof(value(model.qnetwork.mu_layer.w))

    c = 0.0
    hdec = get_hdec(model, x)
    for t = 1:model.T
        # update xhat and then read
        xhat = x .- sigm.(c)
        rt = model.read_layer(x, xhat, hdec)

        # encoder
        model.encoder(vcat(rt, hdec); hidden=model.encoder_hidden)
        henc, cenc = model.encoder_hidden
        henc = reshape(henc, size(henc)[1:2])

        # qnetwork
        z, mu, logsigma, sigma = model.qnetwork(henc)

        # decoder
        model.decoder(z; hidden=model.decoder_hidden)
        hdec, cdec = model.decoder_hidden
        hdec = reshape(hdec, size(hdec)[1:2])

        # write and update draw output
        wt = model.write_layer(hdec)
        c = c .+ wt
        push!(output, mu, logsigma, sigma, c)
    end
    return output
end


# generate
function (model::DRAW)(batchsize::Int)
    empty!(model.encoder_hidden)
    empty!(model.decoder_hidden)
    output = DRAWOutput()
    for t = 1:model.T
        z = sample_noise(model, batchsize)
        c = t == 1 ? 0.0 : output.cs[end]
        model.decoder(z; hidden=model.decoder_hidden)
        hdec, cdec = model.decoder_hidden
        hdec = reshape(hdec, size(hdec)[1:2])
        wt = model.write_layer(hdec)
        c = c .+ wt
        push!(output, c)
    end

    for i = 1:length(output.cs)
        output.cs[i] = Array(sigm.(output.cs[i]))
    end
    return output
end


function get_hdec(model::DRAW, x)
    h = model.state0
    batchsize = size(x, 2)
    h = h .+ fill!(similar(value(h), length(h), batchsize), 0)
    return h
end


function sample_noise(model::DRAW, batchsize::Int)
    return sample_noise(model.qnetwork, batchsize)
end


function binary_cross_entropy(x, x̂)
    F = _etype
    s = @. x * log(x̂ + F(1e-10)) + (1-x) * log(1 - x̂ + F(1e-10))
    return -mean(s)
end


function loss(model::DRAW, x)
    output = model(x)
    xhat = sigm.(output.cs[end])
    Lx = binary_cross_entropy(x, xhat) * model.A * model.B
    kl_terms = []
    Lz = 0.0
    for t = 1:model.T
        mu_2 = output.mus[t] .* output.mus[t]
        sigma_2 = output.sigmas[t] .* output.sigmas[t]
        logsigma = output.logsigmas[t]
        kl = 0.5 * sum((mu_2 + sigma_2-2logsigma), dims=1) .- 0.5
        push!(kl_terms, kl)
    end
    kl_sum = reduce(+, kl_terms)
    Lz = mean(kl_sum)
    return Lx + Lz
end


function init_opt!(model::DRAW, optimizer="Adam()")
    for par in params(model)
        par.opt = eval(Meta.parse(optimizer))
    end
end


function train!(model::DRAW, x)
    J = @diff loss(model, x)
    for par in params(model)
        g = grad(J, par)
        update!(value(par), g, par.opt)
    end
    return value(J)
end


function epoch!(model::DRAW, data)
    atype = typeof(value(model.qnetwork.mu_layer.w))
    lossval = 0.0
    iter = 0
    for (x, y) in data
        J = train!(model, atype(reshape(x, 784, size(x,4))))
        lossval += J
        iter += 1
    end
    return lossval/iter
end


function validate(model::DRAW, data)
    atype = typeof(value(model.qnetwork.mu_layer.w))
    lossval = 0.0
    iter = 0
    for (x, y) in data
        J = loss(model, atype(reshape(x, 784, size(x,4))))
        lossval += J
        iter += 1
    end
    return lossval/iter
end


function parse_options(args)
    s = ArgParseSettings()
    s.description = "DRAW model on MNIST."

    @add_arg_table s begin
        # ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}");
        #  help="array and float type to use")
        ("--batchsize"; arg_type=Int; default=50; help="batch size")
        ("--zdim"; arg_type=Int; default=10; help="noise dimension")
        ("--encoder_dim"; arg_type=Int; default=256; help="hidden units")
        ("--decoder_dim"; arg_type=Int; default=256; help="hidden units")
        ("--epochs"; arg_type=Int; default=20; help="# of training epochs")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
        ("--gridsize"; arg_type=Int; nargs=2; default=[9,9])
        ("--gridscale"; arg_type=Float64; default=2.0)
        ("--optim"; default="Adam(;beta1=0.5, gclip=5.0)")
        ("--loadfile"; default=nothing; help="file to load trained models")
        ("--outdir"; default=nothing; help="output dir for models/generations")
        ("--A"; arg_type=Int; default=28)
        ("--B"; arg_type=Int; default=28)
        ("--N"; arg_type=Int; default=28)
        ("--T"; arg_type=Int; default=10)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    # o[:atype] = eval(parse(o[:atype]))
    if o[:outdir] != nothing
        o[:outdir] = abspath(o[:outdir])
    end
    return o
end


function main(args)
    o = parse_options(args)
    o[:seed] > 0 && Knet.setseed(o[:seed])

    model = DRAW(
        o[:A], o[:B], o[:N], o[:T], o[:encoder_dim],
        o[:decoder_dim], o[:zdim])
    init_opt!(model, o[:optim])
    dtrn,dtst = mnistdata(xtype=Array{Float32})

    bestloss = Inf
    for epoch = 1:o[:epochs]
        trnloss = epoch!(model, dtrn)
        tstloss = validate(model, dtst)
        datetime = now()
        @show datetime, epoch, trnloss, tstloss
        if tstloss < bestloss
            bestloss = tstloss
        end
    end
end
