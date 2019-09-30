function train!(model::Network, x, y)
    values = []
    J = @diff loss(model, x, y; loss_values=values)
    for par in params(model)
        g = grad(J, par)
        update!(value(par), g, par.opt)
    end
    return (sum(values), values[1], values[2])
end


function epoch!(model::Network, data)
    atype = typeof(value(model.qnetwork.mu_layer.w))
    Lx = Lz = 0.0f0
    iter = 0
    @time for (x, y) in data
        J1, J2 = train!(model, atype(reshape(x, 784, size(x,4))), y)
        Lx += J1
        Lz += J2
        iter += 1
    end
    lossval = Lx+Lz
    return lossval/iter, Lx/iter, Lz/iter
end


function validate(model::Network, data)
    atype = typeof(value(model.qnetwork.mu_layer.w))
    Lx = Lz = 0.0f0
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
        ("--gridscale"; arg_type=Float32; default=2.0f0)
        ("--optim"; default="Adam(;beta1=0.5f0, gclip=5.0f0)")
        ("--loadfile"; default=nothing; help="file to load trained models")
        ("--outdir"; default=nothing; help="output dir for models/generations")
        ("--A"; arg_type=Int; default=28)
        ("--B"; arg_type=Int; default=28)
        ("--N"; arg_type=Int; default=5)
        ("--T"; arg_type=Int; default=10)
        ("--valid"; action=:store_true)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    # o[:atype] = eval(parse(o[:atype]))
    if o[:outdir] != nothing
        o[:outdir] = abspath(o[:outdir])
    end
    return o
end


function train(args)
    println("script started"); flush(stdout)
    o = parse_options(args)
    o[:seed] > 0 && Knet.seed!(o[:seed])
    println("options parsed"); flush(stdout)
    display(o)

    model = Network(
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
        tstloss = o[:valid] ? validate(model, dtst) : (.0f0, .0f0, .0f0)
        report(epoch, trnloss, tstloss)
        if tstloss[1] < bestloss
            bestloss = tstloss[1]
        end
    end
    return model
end
