# Meta Reinforcement learning

Simple Julia implementation of the bandit task and two-tap tasks from 'Prefrontal cortex as a meta-reinforcement learning system' (Wang & Kurth-Nelson et al. 2018).

Usage:

`julia bandit_train.jl` #train RNN on the bandit task\\
`julia bandit_anal.jl` #plot simple results

`julia twostep_train.jl` #train RNN on the twostep task\\
`julia twostep_anal.jl` #plot simple results

The default training procedure uses 10,000 'episodes' (as defined by Wang & Kurth-Nelson) and takes ~6-7 minutes to train on CPU for each task. This is generally sufficient for convergence but can also be increased to improve the convergence probability.



