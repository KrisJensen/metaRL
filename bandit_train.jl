include("bandit_utils.jl")
using Flux, Statistics, Random, Distributions, StatsFuns, Zygote, BSON

### initialize model ###
m = Chain(LSTM(Nin, Nhidden), Dense(Nhidden, Nout)) 

### check that our evaluation functions work ###
L, ys, rews, as, ps, hs, states = run_episode(m)
model_loss((), ())
model_eval()

### set training parameters ###
data = repeat([((), ())], 100); #100 episodes per epoch
prms = Flux.params(m);
opt= ADAM(5e-4)
evalcb = () -> @show model_eval()

### train model ###
t0 = time()
for i = 1:100
    println("epoch ", i, "  t=", round(time()-t0, digits = 1))
    Flux.train!(model_loss, prms, data, opt, cb = Flux.throttle(evalcb, 3))
end

### save model ###
using BSON: @save
@save "models/bandit.bson" m
