using Knet

# terms
# g, gx, gy: grid centers
# NxN: gaussian filter size
# delta: stride
# AxB: image size
# rng: randomly sampled matrix


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


function qnet(henc)
end


function reconstruct(w,r,x,o)
    mus = []; logsigmas = []; sigmas = []; cs = []

    c = 0.0
    xhat = x - sigm(c)
    rt = draw_read(x, o)
    henc, cenc = rnnforw(r,w,rt; hy=true, cy=true)
    z, mu, logsigma, sigma = qnet(henc)
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
end


function generate()
end


function loss(w,x,mus,sigmas,logsigmas,o)
    xhat = sigm.(reconstruct)
end
