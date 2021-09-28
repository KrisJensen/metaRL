include("twostep_utils.jl")
using LinearAlgebra, PyPlot, MultivariateStats, BSON

### load model ###
using BSON: @load
@load "models/twostep.bson" m

##### lets do some evaluation ####

function common(action, state)
    if (action == 1 && Bool(state[2]))
        return 1
    elseif (action == 2 && Bool(state[3]))
        return 1
    else
        return 0
    end
end

test_as = zeros(T, 0)
test_rews = zeros(T, 0)

types = []
stays = []
for i = 1:1000
    L, ys, rews, as, ps, hs, states = run_episode(m, hidden = true)
    test_as = [test_as as]
    test_rews = [test_as as]

    for t = 11:2:(T-2)
        a = as[t] #action on THIS trial
        state = states[:, t+1] #NEXT state
        comm = common(a, state) #was the transition common?
        rew = rews[t+1] #NEXT reward (after the NEXT state)
        type = 2*(1-rew) + 1 - 1*comm #0 (rew common), 1 (rew uncommon), 2 (unrew common), 3 (unrew uncommon)
        stay = (a == as[t+2]) #was my next action from state 1 the same?
        types = [types; type]
        stays = [stays; stay]
    end

end

stayprobs = [mean(stays[types .== i]) for i = 0:3]
println(stayprobs)

f = figure()
bar([-2; -1; 1; 2], stayprobs, color = ["b"; "r"; "b"; "r"])
bar([0], [0], color = "r")
xlabel("trial type", fontsize = 12)
ylabel("stay probability", fontsize = 12)
ylim(0.5, 1)
legend(["common", "uncommon"])

ax = f.get_axes()[1]
ax.spines["top"].set_visible(false)
ax.spines["right"].set_visible(false)
xticks([-1.5, 1.5], ["rewarded", "unrewarded"])
savefig("figs/test_twostep.png", bbox_inches = "tight")
close()