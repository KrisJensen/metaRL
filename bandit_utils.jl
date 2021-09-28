include("A2C_utils.jl")
using Flux, Statistics, Random, Distributions, StatsFuns, Zygote

T = 100
Nhidden = 50
Nstates = 1
Nstate_rep = 1
Naction = 2
Nin = Naction+1+1 #no state information for this agent, just actions, reward, time
Nout = Naction+1
βv = 0.05f0
βe = 0.05f0
γ = 0.9f0

function forward(m, x)
    #this just adds a softmax to the policy output
    y = m(x)
    logπ = y[1:2]
    logπ = logπ .- StatsFuns.logsumexp(logπ) #softmax
    return [logπ; y[3]]
end

function env(y, t, state, ps)
    ### input the agent output; output action, reward, and new input
    πt = exp.(y[1:2]) #probability of pulling arms
    d = Binomial(1, Float64(πt[2])) #probability of second arm
    a = (rand(d)+1) # 0 -> 1, 1 -> 2
    rew = rand(Binomial(1, Float64(ps[a]))) #draw reward from corresponding arm
    ahot = zeros(2)
    ahot[a] = 1.
    x = [ahot; rew; t] #input is action, reward, timestep
    ps = ps #reward probabilities do not change
    s_new = zeros(1) #state is empty
    return Int(a), Float32(rew), Float32.(x), Float32.(ps), Float32.(s_new)
end
Zygote.@nograd env #don't take gradients of this

function initialize(ps, state)
    if maximum(ps) == 0 #if not provided
        p = rand()*0.95+0.025
        ps = [p; 1-p]
    end
    state = Float32.(zeros(1)) #no state
    return Float32.(ps), Float32.(state)
end
Zygote.@nograd initialize #don't take gradients of this

function model_eval()
    #evaluation function; compute mean reward (random is 0.5; oracle is 0.745)
    means = []
    all_as = []
    all_ps = [[i/50.; 1-i/50.] for i = 1:49]
    for ps = all_ps
        L, ys, rews, as, ps = run_episode(m, ps = ps, hidden = false)
        means = [means; mean(rews)]
        all_as = [all_as; mean(as)]
    end
    return mean(means), mean(all_as)
end