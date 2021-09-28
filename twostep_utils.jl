include("A2C_utils.jl")
using Flux, Statistics, Random, Distributions, StatsFuns, Zygote

T = 100
Nhidden = 50
Nstates = 3
Nstate_rep = 3
Naction = 2
Nin = Naction+1+1+Nstates #2 actions, 1 rew, 1 time, 3 states
Nout = Naction+1 #action 1, action 2, value function
βv = 0.05f0
βe = 0.05f0
γ = 0.9f0

### specify environment function per task ###
function env(y, t, state, ps)
    ### input the agent output; output action, reward, and new input
    πt = exp.(y[1:2]) #probability of actions
    d = Binomial(1, Float64(πt[1])) #probability of action 1
    a = (2 - rand(d)) # 1 -> 1, 0 -> 2
    
    ahot = zeros(2)
    if Bool(state[1]) #in initial state; move to new state
        prew = 0 #no reward
        p1 = [0.8; 0.2][a] #probability of going to state 2/3 is 0.8/0.2 for action 1 and vice versa for action 2
        s_new = [0; 0; 0]
        s = 3-rand(Binomial(1, p1)) #0->3, 1->2
        s_new[s] = 1
        #println(s, s_new)
        t = 0
        if rand() < 0.025 ps = [ps[2]; ps[1]] end #flip reward
        ahot[a] = 1.
    else
        if Bool(state[2]) prew = ps[1] else prew = ps[2] end #action-independent reward probability
        s_new = [1; 0; 0] #back to initial state
        t = 1
    end

    rew = rand(Binomial(1, Float64(prew))) #draw reward probabilistically
    x = [ahot; rew; t; s_new] #input is action, reward, timestep, state
    return Int(a), Float32(rew), Float32.(x), Float32.(ps), Float32.(s_new)
end
Zygote.@nograd env #don't take gradients of this

### task-specific initialization ###
function initialize(ps, state)
    if maximum(ps) <= 0
        if rand() < 0.5
            ps = [0.9; 0.1]
        else
            ps = [0.1; 0.9]
        end
    end
    if maximum(state) <= 0
        state = [1; 0; 0]
    end
    return Float32.(ps), Float32.(state)
end
Zygote.@nograd initialize #don't take gradients of this

### task specific evaluation/progress function ####
function model_eval()
    #evaluation function; compute mean reward (random is 0.5; oracle is 0.74)
    means = []
    all_as = []
    for i = 1:50
        L, ys, rews, as, ps = run_episode(m, hidden = false)
        means = [means; 2*mean(rews)] #compute mean reward for second stage
        all_as = [all_as; mean(as)]
    end
    return mean(means), mean(all_as)
end