using Knet
import Knet: Knet, minibatch, params, train!
using Sloth

using Images
using ArgParse
using ImageMagick
using JLD2
using MosaicViews

using Statistics, Random, Dates
import Base: push!, empty!


SEED = -1

include(Knet.dir("data","mnist.jl"))


_etype = gpu() >= 0 ? Float32 : Float64
_atype = gpu() >= 0 ? KnetArray{_etype} : Array{_etype}


square(x) = x .* x


struct ReadNoAttention
end


function (l::ReadNoAttention)(x, xhat, hdec;
                              Fx=nothing, Fy=nothing, gamma=nothing)
    vcat(x, xhat)
end


WriteNoAttention = Linear


struct QNet
    mu_layer
    logsigma_layer
end


function (l::QNet)(henc)
    mu = l.mu_layer(henc)
    logsigma = l.logsigma_layer(henc)
    sigma = exp.(logsigma)
    if SEED == -1
        noise = randn!(similar(mu))
    else
        Knet.seed!(SEED)
        noise = randn!(similar(mu))
    end
    sampled = mu .+ noise .* sigma
    return (sampled, mu, logsigma, sigma)
end


function QNet(input_dim::Int, output_dim::Int, atype=_atype, init=xavier)
    mu_layer = Linear(input_dim, output_dim; atype=atype, init=init)
    logsigma_layer = Linear(input_dim, output_dim; atype=atype, init=init)
    return QNet(mu_layer, logsigma_layer)
end


function sample_noise(q::QNet, batchsize::Int; generation=false)
    zdim = size(value(q.mu_layer.w), !generation ? 2 : 1)
    if SEED == -1
        z = randn(zdim, batchsize)
    else
        Knet.seed!(SEED)
        z = randn(zdim, batchsize)
    end
    atype = typeof(value(q.mu_layer.w))
    return convert(atype, z)
end


#
# Attention-related modules
#


mutable struct AttentionWindow
    linear
    A
    B
    N
end


function (l::AttentionWindow)(hdec)
    params = l.linear(hdec)
    gx, gy, logsigma2, logdelta, loggamma = map(i->params[i,:], size(params,1))
    gx = ((l.A + 1) / 2) .* (gx .+ 1)
    gy = ((l.B + 1) / 2) .* (gy .+ 1)
    delta = (max(l.A, l.B) - 1) / (l.N - 1) .* exp.(logdelta)
    sigma2 = exp.(logsigma2)
    gamma = exp.(loggamma)

    Fx, Fy = filterbank(gx, gy, sigma2, delta, l.A, l.B, l.N)
    return Fx, Fy, gamma
end


function AttentionWindow(input_dim::Int, A::Int, B::Int, N::Int; atype=_atype)
    linear = Linear(input_dim, 5; atype=atype)
    return AttentionWindow(linear, A, B, N)
end


function filterbank(gx, gy, sigma2, delta, A, B, N)
    atype = typeof(gx) <: KnetArray ? KnetArray : Array
    etype = eltype(gx)
    batchsize = size(gx, 2)

    rng = atype{etype}(1:N)
    mu_x = compute_mu(gx, rng, delta, N)
    mu_y = compute_mu(gy, rng, delta, N)

    a = reshape(atype{etype}(1:A), 1, 1, A)
    b = reshape(atype{etype}(1:B), 1, 1, B)

    mu_x = reshape(mu_x, 1, size(mu_x)...)
    mu_y = reshape(mu_y, 1, size(mu_y)...)
    sig2 = reshape(sigma2, 1, 1, length(sig2))

    Fx = filterbank_matrices(a, mu_x, sig2)
    Fy = filterbank_matrices(b, mu_y, sig2)

    return Fx, Fy
end


function filterbank_matrices(xs, mu, sigma2; epsilon=1e-9)
    y = xs .- mu
    y = y ./ 2sigma2
    y = exp.(-y .* y)
    y = y ./ (sum(y, dims=1) .+ epsilon)
end


function compute_mu(g, rng, delta, N)
    batchsize = size(g, 2)
    rng_t = hcat(map(i->rng, 1:batchsize)...)
    # g_t = vcat(map(i->delta, 1:N)..)
    # delta_t = vcat(map(i->delta, 1:N)...)
    tmp = (rng_t .- 0.5 .* (1+N)) .* delta
    mu = tmp + g
end


function filter_image(image, Fx, Fy, gamma, A, B, N)
    batchsize = size(image, 2)
    Fxt = permutedims(Fx, (2,1,3))
    img = reshape(img, A, B, batchsize)
    glimpse = bmm(Fy, bmm(img, Fxt))
    glimpse = reshape(glimpse, N, N, batchsize)
    return glimpse .* gamma
end


mutable struct ReadAttention
    window
end


function (l::ReadAttention)(x, xhat, hdec)
    Fx, Fy, gamma = l.window(hdec)
    A, B, N = l.window.A, l.window.B, l.window.N
    xnew = filter_image(x, Fx, Fy, gamma, A, B, N)
    xhatnew = filter_image(xhat, Fx, Fy, gamma, A, B, N)
    return vcat(xnew, xhatnew)
end


mutable struct WriteAttention
    window
    linear
end


function (l::WriteAttention)(Fx, Fy, gamma, hdec)
    A, B, N = l.window.A, l.window.B, l.window.N
    batchsize = size(hdec, 2)
    w = l.linear(hdec)
    w = reshape(w, N, N, batchsize)
    Fx, Fy, gamma = l.window(hdec)
    Fyt = permutedims(Fy, 2, 1, 3)
    wr = bmm(Fyt, bmm(w, Fx))
    wr = reshape(wr, A*B, batchsize)
    return wr ./ gamma
end


function WriteAttention(window, decoder_dim, N; atype=_atype)
    linear = Linear(decoder_dim, N*N; atype=atype)
    return WriteAttention(window, linear)
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
    window
    read_layer
    write_layer
    embed_layer
    qnetwork
    encoder
    decoder
    encoder_hidden
    decoder_hidden
    state0
end


function DRAW(A, B, N, T, encoder_dim, decoder_dim, noise_dim;
              embed=0, nclass=10, atype=_atype, attention=true)
    imgsize = A*B

    window = attention ? AttentionWindow(decoder_dim, A, B, N) : nothing
    read_layer = attention ? ReadAttention(window) : ReadNoAttention()
    write_layer =
        if attention
            WriteAttention(window, decoder_dim, N; atype=atype)
        else
            WriteNoAttention(decoder_dim, imgsize; atype=atype)
        end
    embed_layer = embed == 0 ? nothing : Embedding(nclass, embed; atype=atype)
    qnetwork = QNet(decoder_dim, noise_dim, atype)
    encoder = RNN(2imgsize+decoder_dim, encoder_dim; dataType=_etype) # FIXME: adapt to attn
    decoder = RNN(noise_dim+embed, decoder_dim; dataType=_etype)
    encoder_hidden = []
    decoder_hidden = []
    state0 = atype(zeros(decoder.hiddenSize, 1))

    return DRAW(
        A,
        B,
        N,
        T,
        window,
        read_layer,
        write_layer,
        embed_layer,
        qnetwork,
        encoder,
        decoder,
        encoder_hidden,
        decoder_hidden,
        state0
    )
end


function DRAW(N, T, encoder_dim, decoder_dim, noise_dim;
              embed=0, nclass=10, atype=_atype)
    A = B = N
    return DRAW(A, B, N, T, encoder_dim, decoder_dim, noise_dim;
                embed=embed, atype=_atype)
end


# reconstruct
function (model::DRAW)(x, y=nothing; cprev=_atype(zeros(size(x))))
    empty!(model.encoder_hidden)
    empty!(model.decoder_hidden)
    output = DRAWOutput()
    atype = typeof(value(model.qnetwork.mu_layer.w))

    hdec = get_hdec(model, x)
    hinit = reshape(hdec, size(hdec,1), size(hdec,2), 1)
    push!(model.decoder_hidden, hinit, hinit)
    for t = 1:model.T
        # update xhat and then read
        xhat = x - sigm.(cprev)
        rt = model.read_layer(x, xhat, hdec)

        # encoder
        input = vcat(rt, hdec)
        input = reshape(input, size(input, 1), size(input, 2), 1)
        model.encoder(input; hidden=model.encoder_hidden)
        henc, cenc = model.encoder_hidden
        henc = reshape(henc, size(henc)[1:2])

        # qnetwork
        z, mu, logsigma, sigma = model.qnetwork(henc)

        input = z
        if model.embed_layer != nothing
            y == nothing && error("You should pass labels also.")
            input = vcat(z, model.embed_layer(y))
        end

        # decoder
        model.decoder(input; hidden=model.decoder_hidden)
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
function (model::DRAW)(batchsize::Int, y=nothing)
    empty!(model.encoder_hidden)
    empty!(model.decoder_hidden)
    output = DRAWOutput()
    for t = 1:model.T
        z = sample_noise(model, batchsize)
        c = t == 1 ? 0.0 : output.cs[end]

        input = z
        if model.embed_layer != nothing
            y == nothing && error("You should pass labels also.")
            input = vcat(z, model.embed_layer(y))
        end

        model.decoder(input; hidden=model.decoder_hidden)
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
    return sample_noise(model.qnetwork, batchsize; generation=true)
end


function binary_cross_entropy(x, x̂)
    F = _etype
    s = @. x * log(x̂ + F(1e-8)) + (1-x) * log(1 - x̂ + F(1e-8))
    return -mean(s)
end


function loss(model::DRAW, x, y=nothing; loss_values=[])
    output = model(x, y)
    xhat = sigm.(output.cs[end])
    Lx = binary_cross_entropy(x, xhat) * model.A * model.B
    kl_terms = []
    for t = 1:model.T
        mu_2 = square(output.mus[t])
        sigma_2 = square(output.sigmas[t])
        logsigma = output.logsigmas[t]
        kl = 0.5 * sum((mu_2 + sigma_2-2logsigma), dims=1) .- 0.5# *model.T # FIXME: dimension kontrol
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


function train!(model::DRAW, x, y)
    values = []
    J = @diff loss(model, x, y; loss_values=values)
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
        J1, J2 = train!(model, atype(reshape(x, 784, size(x,4))), y)
        Lx += J1
        Lz += J2
        iter += 1
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
        loss(model, atype(reshape(x, 784, size(x,4))), y; loss_values=values)
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
        ("--batchsize"; arg_type=Int; default=64; help="batch size")
        ("--zdim"; arg_type=Int; default=10; help="noise dimension")
        ("--encoder"; arg_type=Int; default=256; help="hidden units")
        ("--decoder"; arg_type=Int; default=256; help="hidden units")
        ("--embed"; arg_type=Int; default=100; help="condition dimension")
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
    display(o)

    model = DRAW(
        o[:A], o[:B], o[:N], o[:T], o[:encoder],
        o[:decoder], o[:zdim]; embed=o[:embed])
    println("model initialized"); flush(stdout)
    init_opt!(model, o[:optim])
    println("optimization parameters initiazlied"); flush(stdout)
    dtrn, dtst = mnistdata(xtype=Array{Float32})
    println("data loaded"); flush(stdout)

    bestloss = Inf
    for epoch = 1:o[:epochs]
        trnloss = epoch!(model, dtrn)
        tstloss = validate(model, dtst)
        report(epoch, trnloss, tstloss)
        if tstloss[1] < bestloss
            bestloss = tstloss[1]
        end
    end
    return model
end


function report(epoch, trn, tst)
    trnloss, trnLx, trnLz = trn
    tstloss, tstLx, tstLz = tst
    datetime = now()
    print("epoch=$epoch, trn=$trnloss (Lx=$trnLx, Lz=$trnLz)")
    println(", tst=$tstloss (Lx=$tstLx, Lz=$tstLz)")
    flush(stdout)
end
