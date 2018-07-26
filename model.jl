using Knet

# terms
# g, gx, gy: grid centers
# NxN: gaussian filter size
# delta: stride
# AxB: image size
# rng: randomly sampled matrix


include(Pkg.dir("Knet","examples","variational-autoencoder", "vae_mnist.jl"))

function filterbank(gx, gx, sigma2, delta, A, B)
    rng = randn!(similar(mu))
    mu_x = compute_mu(gx, rng, delta)
    mu_y = compute_mu(gy, rng, delta)

    atype = typeof(AutoGrad.getval(gx)) <: KnetArray ? KnetArray : Array
    etype = atype <: KnetArray ? Float32 : Float64
    a = convert(atype{etype}, reshape(collect(1:A), A, 1))
    b = convert(atype{etype}, reshape(collect(1:B), B, 1))

    Fx = get_filterbank_matrices(a, mu_x, sigma2)
    Fy = get_filterbank_matrices(b, mu_y, sigma2)

    return Fx, Fy
end


function get_filterbank_matrices(a, mu_x, sigma2, epsilon=1e-9)
    F = -((a .- mu_x) / 2sigma2).^2
    F = F/(sum(F,2)+epsilon)
end


function compute_mu(g, rng, delta, N)
    mu = (rng .- N / 2 - 0.5) * delta .+ g
    mu = reshape(mu, N, 1)
end


function att_window(w, hdec)
end


function qnet(w,henc)
    mu = w[:wmu] * henc .+ w[:bmu]
    logsigma = w[:wlogsigma] * henc .+ w[:blogsigma]
    sigma = exp.(logsigma)
    noise = randn!(similar(mu))
    return mu .+ noise .* sigma
end


function reconstruct(w,r,x,o)
    mus = []; logsigmas = []; sigmas = []; cs = []

    c = 0.0
    xhat = x - sigm(c)
    rt = draw_read(x, o)
    henc, cenc = rnnforw(r,w,rt; hy=true, cy=true)
    z, mu, logsigma, sigma = qnet(w,henc)
    push!(mus, mu); push!(logsigmas, logsigma); push!(sigmas, sigma)
    hdec, cdec = rnnforw(r,w,z; hy=true, cy=true)
    wt = draw_write(hdec)
    c = c .+ wt
    push!(cs, c)

    for t=2:o[:nsteps]
        xhat = x - sigm(c)
        rt = draw_read(x, xhat, hdec)
        henc, cenc = rnnforw(r, w, rt, henc, cenc; hy=true, cy=true)
        z, mu, logsigma, sigma = qnet(henc)
        push!(mus, mu); push!(logsigmas, logsigma); push!(sigmas, sigma)
        hdec, cdec = rnnforw(r, w, z, hdec, cdec; hy=true, cy=true)
        wt = draw_write(hdec)
        c = c .+ wt
        push!(cs, c)
    end

    return mus, logsigmas, sigmas, cs
end


function generate()
end


function loss(w,r,x,o)
    A, B, T = o[:A], o[:B], o[:T]
    mus, logsigmas, sigmas, cs = reconstruct(w, r, x, o)
    xhat = sigm.(cs[end])
    Lx = VAE.binary_cross_entropy(x, xhat) * A * B
    Lz = 0
    for t = 1:T
        mu_2 = mus[t] * mus[t]
        sigma_2 = sigmas[t] * sigmas[t]
        logsigma = logsigmas[t]
        Lz += 0.5 * sum(mu_2 * sigma_2-2logsigma, 1) - 0.5T
    end
    Lz = mean(Lz)
    return Lx + Lz
end
