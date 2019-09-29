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
    ps = l.linear(hdec)
    hsize, batchsize = size(ps)
    gx, gy, logsigma2, logdelta, loggamma = map(
        i->reshape(ps[i, :], 1, batchsize) , 1:hsize)
    gx = div(l.A+1, 2) .* (gx .+ 1)
    gy = div(l.B+1, 2) .* (gy .+ 1)
    delta = div(max(l.A, l.B) - 1, (l.N - 1)) .* exp.(logdelta)
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
    atype = typeof(value(gx)) <: KnetArray ? KnetArray : Array
    etype = eltype(gx)
    batchsize = size(gx, 2)

    rng = reshape(atype{etype}(0:N-1), N, 1)
    mu_x = compute_mu(gx, rng, delta, N)
    mu_y = compute_mu(gy, rng, delta, N)

    a = reshape(atype{etype}(0:A-1), A, 1, 1)
    b = reshape(atype{etype}(0:B-1), B, 1, 1)

    mu_x = reshape(mu_x, 1, size(mu_x)...)
    mu_y = reshape(mu_y, 1, size(mu_y)...)
    sig2 = reshape(sigma2, 1, 1, length(sigma2))

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
    tmp = (rng .- 0.5f0 .- div(N, 2)) .* delta
    mu = tmp .+ g
end


function filter_image(image, Fxt, Fy, gamma, A, B, N)
    batchsize = size(image, 2)
    img = reshape(image, A, B, batchsize)
    glimpse = bmm(bmm(Fxt, img), Fy)
    glimpse = reshape(glimpse, N, N, batchsize)
    out = glimpse .* reshape(gamma, 1, 1, batchsize)
    out = reshape(out, div(length(out), batchsize), batchsize)
end


mutable struct ReadAttention
    window
end


function (l::ReadAttention)(x, xhat, hdec)
    Fx, Fy, gamma = l.window(hdec)
    A, B, N = l.window.A, l.window.B, l.window.N
    Fxt = permutedims(Fx, (2, 1, 3))
    xnew = filter_image(x, Fxt, Fy, gamma, A, B, N)
    xhatnew = filter_image(xhat, Fxt, Fy, gamma, A, B, N)
    return vcat(xnew, xhatnew)
end


mutable struct WriteAttention
    window
    linear
end


function (l::WriteAttention)(hdec)
    A, B, N = l.window.A, l.window.B, l.window.N
    batchsize = size(hdec, 2)
    w = l.linear(hdec)
    w = reshape(w, N, N, batchsize)
    Fx, Fy, gamma = l.window(hdec)
    Fyt = permutedims(Fy, (2, 1, 3))
    wr = bmm(bmm(Fx, w), Fyt)
    wr = reshape(wr, A*B, batchsize)
    return wr ./ gamma
end


function WriteAttention(window, decoder_dim, N; atype=_atype)
    linear = Linear(decoder_dim, N*N; atype=atype)
    return WriteAttention(window, linear)
end
