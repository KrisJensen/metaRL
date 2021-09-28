using Pkg
Pkg.activate(@__DIR__)
using Flux, Statistics, Random, Distributions, StatsFuns, Zygote, CUDA, ChainRulesCore

function ChainRulesCore.rrule(::typeof(cpu), x::CuArray{Float32})
    pullback(Δ::Array{Float32}) = NoTangent(), cu(Δ)
    return cpu(x), pullback
end
use_cuda = false #cpu by default

function forward(m, x)
    #this just adds a softmax to the policy output
    if use_cuda
        y = m( Zygote.@ignore cu(x) ) |> cpu #RNN step
    else
        y = m(x)
    end
    logπ = y[1:Naction]
    logπ = logπ .- StatsFuns.logsumexp(logπ) #softmax
    return [logπ; y[Naction+1]]
end

function calc_deltas(rews, Vs)
    #compute RPEs
    R = 0
    δs = []
    for t = 0:(T-1) #compute RPE
        R = rews[T-t] + γ*R #Reward
        δs = [R - Vs[T-t]; δs] #error
    end
    return Float32.(δs)
end
Zygote.@nograd calc_deltas #don't take gradients of this

function run_episode(m; ps = zeros(2), state = zeros(3), hidden = true)
    ### initialize reward probabilities and state ###
    ps, state = initialize(ps, state)

    #arrays for storing output
    #ys = Matrix{Float32}(undef, Nout,0)
    ys = Matrix{Float32}(undef, Nout,0)
    if hidden
        hs = Matrix{Float32}(undef, Nhidden, 0)
        states = Matrix{Float32}(undef, Nstate_rep, 0) #Nstate_rep is the non-one-hot size
    end
    as = Matrix{Int32}(undef, 1,0)
    rews = Matrix{Float32}(undef, 1,0)

    ### reset model and initialize input!! ###
    Flux.reset!(m)
    x = Float32.(zeros(Nin))
    for t = 1:T
        if hidden
            hs = [hs m[1].state[1]] #store hidden state
            states = [states state] #store state
        end

        y = forward(m, x) #RNN step
        a, rew, x, ps, state = env(y, t, state, ps) #output -> action, reward, next input

        ys = [ys y] #store output
        rews = [rews rew] #store reward
        as = [as a] #store action
    end

    δs = calc_deltas(rews, ys[Naction+1, :])
    L = Float32(0.)
    for t = 1:T
        logπ=ys[1:Naction, t]
        H = -sum(exp.(logπ) .* logπ)
        RPE_term = δs[t]*(ys[as[t], t] + βv*ys[Naction+1, t])
        L -= (RPE_term + βe*H) #loss
    end

    hidden && return L, ys, rews[1,:], as[1,:], ps, hs, states#, xs
    return L, ys, rews[1,:], as[1,:], ps
end

function model_loss((), ())
    #wrapper for Flux that takes empty data
    loss, _, _, _, _ = run_episode(m, hidden = false)
    return loss
end
