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


square(x) = x .* x


function urand(output, input)
    dim = input
    max_value = sqrt(3/dim)
    min_value = -max_value
    w = min_value .+ rand(output, input) .* (max_value-min_value)
end


function initwb(input_dim::Int, output_dim::Int, atype=_atype, init=urand)
    w = param(output_dim, input_dim; init=init, atype=_atype)
    b = param(output_dim, 1; atype=_atype)
    return (w,b)
end


struct Linear
    w
    b
end


(l::Linear)(x) = l.w * x .+ l.b


function Linear(input_dim::Int, output_dim::Int, atype=_atype, init=urand)
    w, b = initwb(input_dim, output_dim, atype, init)
    return Linear(w, b)
end


struct FullyConnected
    w
    b
    activate
end


(l::FullyConnected)(x) = activate.(l.w * x .+ l.b)


function FullyConnected(
    input_dim::Int, output_dim::Int, activate=relu, atype=_atype, init=urand)
    w, b = initwb(input_dim, output_dim, atype, init)
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
    noise = randn!(similar(mu)) # FIXME
    sampled = mu .+ noise .* sigma
    return (sampled, mu, logsigma, sigma)
end


function QNet(input_dim::Int, output_dim::Int, atype=_atype, init=urand)
    mu_layer = Linear(input_dim, output_dim, atype, init)
    logsigma_layer = Linear(input_dim, output_dim, atype, init)
    return QNet(mu_layer, logsigma_layer)
end


function sample_noise(q::QNet, batchsize::Int)
    zdim = size(value(q.mu_layer.w), 2)
    z = randn(zdim, batchsize)
    atype = typeof(value(q.mu_layer.w))
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
    imgsize = A*B
    read_layer = ReadNoAttention()
    write_layer = WriteNoAttention(decoder_dim, imgsize, atype)
    qnetwork = QNet(decoder_dim, noise_dim, atype)
    encoder = RNN(2imgsize+decoder_dim, encoder_dim) # FIXME: adapt to attn
    decoder = RNN(noise_dim, decoder_dim)
    encoder_hidden = []
    decoder_hidden = []
    state0 = atype(zeros(decoder.hiddenSize, 1))

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
        state0
    )
end


function DRAW(N, T, encoder_dim, decoder_dim, noise_dim, atype=_atype)
    A = B = N
    return DRAW(A, B, N, T, encoder_dim, decoder_dim, noise_dim, atype=_atype)
end


# reconstruct
function (model::DRAW)(x; cprev=_atype(zeros(size(x))))
    empty!(model.encoder_hidden)
    empty!(model.decoder_hidden)
    output = DRAWOutput()
    atype = typeof(value(model.qnetwork.mu_layer.w))

    hdec = get_hdec(model, x)
    push!(model.decoder_hidden, hdec, hdec)
    for t = 1:model.T
        # update xhat and then read
        xhat = x - sigm.(cprev)
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
        c = cprev + model.write_layer(hdec)
        push!(output, mu, logsigma, sigma, c)
        cprev = output.cs[end]
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
    s = @. x * log(x̂ + F(1e-8)) + (1-x) * log(1 - x̂ + F(1e-8))
    return -mean(s)
end


function loss(model::DRAW, x; loss_values=[])
    output = model(x)
    xhat = sigm.(output.cs[end])
    Lx = binary_cross_entropy(x, xhat) * model.A * model.B
    kl_terms = []
    for t = 1:model.T
        mu_2 = square(output.mus[t])
        sigma_2 = square(output.sigmas[t])
        logsigma = output.logsigmas[t]
        kl = 0.5 * sum((mu_2 + sigma_2-2logsigma), dims=1) .- 0.5 # FIXME: dimension kontrol
        push!(kl_terms, kl)
    end
    kl_sum = reduce(+, kl_terms) # == sum(kl_terms)
    Lz = mean(kl_sum)
    push!(loss_values, value.(Lx), value.(Lz))
    return Lx + Lz
end


function init_opt!(model::DRAW, optimizer="Adam()")
    for par in params(model)
        par.opt = eval(Meta.parse(optimizer))
    end
end


function train!(model::DRAW, x)
    values = []
    J = @diff loss(model, x; loss_values=values)
    for par in params(model)
        g = grad(J, par)
        update!(value(par), g, par.opt)
    end
    return (sum(values), values[1], values[2])
end


function epoch!(model::DRAW, data)
    atype = typeof(value(model.qnetwork.mu_layer.w))
    Lx = Lz = 0.0
    iter = 0
    for (x, y) in data
        # @show iter
        # flush(stdout)
        J1, J2 = train!(model, atype(reshape(x, 784, size(x,4))))
        Lx += J1
        Lz += J2
        iter += 1

        if (iter-1) % 100 == 0
            println("iter=$iter, Lx=$(Lx/iter), Lz=$(Lz/iter)")
        end
    end
    lossval = Lx+Lz
    return lossval/iter, Lx/iter, Lz/iter
end


function validate(model::DRAW, data)
    atype = typeof(value(model.qnetwork.mu_layer.w))
    Lx = Lz = 0.0
    iter = 0
    values = []
    for (x, y) in data
        loss(model, atype(reshape(x, 784, size(x,4))); loss_values=values)
        J1, J2 = values
        Lx += J1
        Lz += J2
        iter += 1
        empty!(values)
    end
    lossval = Lx + Lz
    return lossval/iter, Lx/iter, Lz/iter
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
    println("script started"); flush(stdout)
    o = parse_options(args)
    o[:seed] > 0 && Knet.seed!(o[:seed])
    println("options parsed"); flush(stdout)

    model = DRAW(
        o[:A], o[:B], o[:N], o[:T], o[:encoder_dim],
        o[:decoder_dim], o[:zdim])
    println("model initialized"); flush(stdout)
    init_opt!(model, o[:optim])
    println("optimization parameters initiazlied"); flush(stdout)
    dtrn, dtst = mnistdata(xtype=Array{Float32})
    println("data loaded"); flush(stdout)

    bestloss = Inf
    for epoch = 1:o[:epochs]
        trnloss = epoch!(model, dtrn)
        tstloss = validate(model, dtst)
        datetime = now()
        @show datetime, epoch, trnloss, tstloss
        flush(stdout)
        if tstloss[1] < bestloss
            bestloss = tstloss[1]
        end
    end
    return model
end
