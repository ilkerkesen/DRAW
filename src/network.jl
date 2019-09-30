mutable struct Network
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
end


function Network(A, B, N, T, encoder_dim, decoder_dim, noise_dim;
              embed=0, nclass=10, atype=_atype, read_attn=true, write_attn=true)
    imgsize = A*B

    window = nothing
    if read_attn || write_attn
        window = AttentionWindow(decoder_dim, A, B, N; atype=atype)
    end

    read_layer = read_attn ? ReadAttention(window) : ReadNoAttention()
    write_layer = nothing
    if write_attn
        write_layer = WriteAttention(window, decoder_dim, N; atype=atype)
    else
        write_layer = WriteNoAttention(decoder_dim, imgsize; atype=atype)
    end

    embed_layer = embed == 0 ? nothing : Embedding(nclass, embed; atype=atype)
    qnetwork = QNet(decoder_dim, noise_dim, atype)
    encoder = RNN(2*N*N+decoder_dim, encoder_dim; dataType=_etype) # FIXME: adapt to attn
    decoder = RNN(noise_dim+embed, decoder_dim; dataType=_etype)
    encoder_hidden = []
    decoder_hidden = []

    return Network(
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
        decoder_hidden
    )
end


function Network(N, T, encoder_dim, decoder_dim, noise_dim;
              embed=0, nclass=10, atype=_atype, read_attn=true, write_atnn=true)
    A = B = N
    Network(A, B, N, T, encoder_dim, decoder_dim, noise_dim;
         embed=embed, atype=_atype, read_attn=read_attn, write_attn=write_attn)
end


# reconstruct
function (model::Network)(x, y=nothing; cprev=_atype(zeros(size(x))))
    empty!(model.encoder_hidden)
    empty!(model.decoder_hidden)
    output = DRAWOutput()
    atype = typeof(value(model.qnetwork.mu_layer.w))

    hdec = get_hdec(model, x)
    hinit = reshape(hdec, size(hdec,1), size(hdec,2), 1)
    model.decoder.h, model.decoder.c = hinit, 0.0f0
    model.encoder.h = model.encoder.c = 0.0f0
    for t = 1:model.T
        # update xhat and then read
        xhat = x - sigm.(cprev)
        rt = model.read_layer(x, xhat, hdec)

        # encoder
        input = vcat(rt, hdec)
        input = reshape(input, size(input, 1), size(input, 2), 1)
        model.encoder(input)
        henc = model.encoder.h
        henc = reshape(henc, size(henc)[1:2])

        # qnetwork
        z, mu, logsigma, sigma = model.qnetwork(henc)

        input = z
        if model.embed_layer != nothing
            y == nothing && error("You should pass labels also.")
            input = vcat(z, model.embed_layer(y))
        end

        # decoder
        model.decoder(input)
        hdec = model.decoder.h
        hdec = reshape(hdec, size(hdec)[1:2])

        # write and update draw output
        c = cprev + model.write_layer(hdec)
        push!(output, mu, logsigma, sigma, c)
        cprev = output.cs[end]
    end
    return output
end


# generate
function (model::Network)(batchsize::Int, y=nothing)
    empty!(model.encoder_hidden)
    empty!(model.decoder_hidden)
    output = DRAWOutput()
    model.decoder.h = model.decoder.c = 0.0f0
    for t = 1:model.T
        z = sample_noise(model, batchsize)
        c = t == 1 ? 0.0f0 : output.cs[end]

        input = z
        if model.embed_layer != nothing
            y == nothing && error("You should pass labels also.")
            input = vcat(z, model.embed_layer(y))
        end

        model.decoder(input)
        hdec = model.decoder.h
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


function get_hdec(model::Network, x)
    num_hidden, batchsize = model.decoder.hiddenSize, size(x, 2)
    h = _atype(zeros(num_hidden, batchsize))
end


function sample_noise(model::Network, batchsize::Int)
    return sample_noise(model.qnetwork, batchsize; generation=true)
end


function binary_cross_entropy(x, x̂)
    F = _etype
    s = @. x * log(x̂ + F(1f-8)) + (1-x) * log(1 - x̂ + F(1f-8))
    return -mean(s)
end


function loss(model::Network, x, y=nothing; loss_values=[])
    output = model(x, y)
    xhat = sigm.(output.cs[end])
    Lx = binary_cross_entropy(x, xhat) * model.A * model.B
    kl_terms = []
    for t = 1:model.T
        mu_2 = square(output.mus[t])
        sigma_2 = square(output.sigmas[t])
        logsigma = output.logsigmas[t]
        kl = 0.5 * sum((mu_2 + sigma_2-2logsigma), dims=1) .- 0.5f0 * model.T
        push!(kl_terms, kl)
    end
    kl_sum = reduce(+, kl_terms) # == sum(kl_terms)
    Lz = mean(kl_sum)
    push!(loss_values, value(Lx), value(Lz))
    return Lx + Lz
end


function init_opt!(model::Network, optimizer="Adam()")
    for par in params(model)
        par.opt = eval(Meta.parse(optimizer))
    end
end
