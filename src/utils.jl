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


function report(epoch, trn, tst=nothing)
    print("[$(now())] epoch=$epoch, trn=$trn")
    if tst != nothing; println(", tst=$tst")
    else; println(); end
    flush(stdout)
end
