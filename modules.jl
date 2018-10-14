using Knet
import Base: push!, empty!


_etype = Float32
_atype = gpu() >= 0 ? KnetArray{_etype} : Array{_etype}
@enum DrawMode RECONSTRUCT=1 GENERATE=2


function initwb(input_dim::Int, output_dim::Int, atype=_atype)
    w = param(output_dim, input_dim; init=randn, atype=_atype)
    b = param(output_dim, 1; atype=_atype)
    return (w,b)
end


struct Linear
    w
    b
end


(l::Linear)(x) = l.w * x .+ b


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
    mu = mu_layer(henc)
    logsigma = logsigma_layer(henc)
    sigma = exp.(logsigma)
    noise = randn!(similar(mu))
    sampled = mu .+ noise .* sigma
    return (sampled, logsigma, sigma)
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


function push!(o::DRAWOutput, mu, logsigma, sigma, cs)
    push!(o.mus, mu)
    push!(o.logsigmas, logsigma)
    push!(o.sigma, sigma)
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
end


function DRAW(A, B, N, T, encoder_dim, decoder_dim, noise_dim, atype=_atype)
    read_layer = Read()
    write_layer = Write(decoder_dim, A*B, atype)
    qnetwork = QNet(encoder_dim, noise_dim, atype)
    encoder = RNN(input_dim, encoder_dim)
    decoder = RNN(input_dim, encoder_dim)

    return DRAW(
        A,
        B,
        N,
        T,
        read_layer,
        write_layer,
        qnetwork,
        encoder,
        decoder
    )
end


function (model::DRAW)(x; mode=RECONSTRUCT::DrawMode)
    output = DRAWOutput()

end


function loss(model::DRAW, x)
    output = model(x)
    xhat = sigm.(cs[end])
    Lx = VAE.binary_cross_entropy(x, xhat) * model.A * model.B
    Lz = 0
    for t = 1:model.T
        mu_2 = mus[t] .* mus[t]
        sigma_2 = sigmas[t] .* sigmas[t]
        logsigma = logsigmas[t]

    end
end
