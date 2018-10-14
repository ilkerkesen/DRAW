function initwb(input_dim::Int, output_dim::Int, atype=Array{Float32})
    w = convert(atype, randn(output_dim, input_dim))
    b = convert(atype, zeros(output_dim, 1))
    return (w,b)
end

abstract type Module
end

struct Linear <: Module
    w
    b
end

(l::Linear)(x) = l.w * x .+ b

function Linear(input_dim::Int, output_dim::Int, atype=Array{Float32})
    w, b = initwb(input_dim, output_dim, atype)
    return Linear(w, b)
end


struct FullyConnected <: Module
    w
    b
    activate
end


(l::FullyConnected)(x) = activate(l.w * x .+ l.b)

function FullyConnected(input_dim::Int, output_dim::Int, atype=Array{Float32})
    w, b = initwb(input_dim, output_dim, atype)
    return FullyConnected(w, b)
end

struct Read <: Module
end

(l::Read)(x, xhat) = vcat(x, xhat)

Write = Linear

struct QNet <: Module
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

function QNet(input_dim::Int, output_dim::Int, atype=Array{Float32})
    mu_layer = Linear(input_dim, output_dim, atype)
    logsigma_layer = Linear(input_dim, output_dim, atype)
    return QNet(mu_layer, logsigma_layer)
end


struct RNN <: Module

end


struct DRAW <: Module
    read_layer
    write_layer
    qnetwork
    encoder
    decoder
end


function DRAW(
    A, B, N, encoder_dim, decoder_dim, noise_dim, atype=Array{Float32})
    read_layer = Read()
    write_layer = Write(decoder_dim, A*B, atype)
    qnetwork = QNet(encoder_dim, noise_dim, atype)
    encoder = RNN(input_dim, encoder_dim, atype)
    decoder = RNN(input_dim, encoder_dim, atype)

    return DRAW(
        read_layer,
        write_layer,
        qnetwork,
        encoder,
        decoder
    )
end

function (model::DRAW)(x)
end
