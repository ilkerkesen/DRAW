using Knet
import Base: push!, empty!
using Statistics, Random
import Knet: params, train!


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


struct Read
end


(l::Read)(x, xhat) = vcat(x, xhat)


Write = Linear


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


struct DRAWOutput
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


struct DRAW
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
end


function DRAW(A, B, N, T, encoder_dim, decoder_dim, noise_dim, atype=_atype)
    encoder_dim = A*B
    read_layer = Read()
    write_layer = Write(decoder_dim, A*B, atype)
    qnetwork = QNet(encoder_dim, noise_dim, atype)
    encoder = RNN(2*N*N+decoder_dim, encoder_dim)
    decoder = RNN(noise_dim, decoder_dim)
    encoder_hidden = []
    decoder_hidden = []

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
        decoder_hidden
    )
end


function DRAW(N, T, encoder_dim, decoder_dim, noise_dim, atype=_atype)
    A = B = N
    return DRAW(A, B, N, T, encoder_dim, decoder_dim, noise_dim, atype=_atype)
end


function sample_noise(q::QNet, batchsize::Int)
    zdim = size(value.(q.mu_layer.w), 2)
    z = randn(zdim, batchsize)
    atype = typeof(value.(q.mu_layer.w))
    return convert(atype, z)
end


function sample_noise(model::DRAW, batchsize::Int)
    return sample_noise(model.qnetwork, batchsize)
end


# reconstruct
function (model::DRAW)(x)
    empty!(model.encoder_hidden)
    empty!(model.decoder_hidden)
    output = DRAWOutput()
    atype = typeof(value.(model.qnetwork.mu_layer.w))

    c = 0.0
    hdec = atype(zeros(model.decoder.hiddenSize, size(x,2)))
    for t = 1:model.T
        # update xhat and then read
        xhat = x .- sigm.(c)
        rt = model.read_layer(x, xhat)

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


function binary_cross_entropy(x, x̂)
    F = _etype
    s = @. x * log(x̂ + F(1e-10)) + (1-x) * log(1 - x̂ + F(1e-10))
    return -mean(s)
end


function loss(model::DRAW, x)
    output = model(x)
    xhat = sigm.(output.cs[end])
    Lx = binary_cross_entropy(x, xhat) * model.A * model.B
    Lz = 0.0
    for t = 1:model.T
        mu_2 = output.mus[t] .* output.mus[t]
        sigma_2 = output.sigmas[t] .* output.sigmas[t]
        logsigma = output.logsigmas[t]
        Lz = Lz .+ 0.5 * sum(mu_2 .* sigma_2-2logsigma, dims=1) .- 0.5 * model.T
    end
    Lz = mean(Lz)
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
    return J
end
