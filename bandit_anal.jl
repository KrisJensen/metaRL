include("bandit_utils.jl")
using LinearAlgebra, PyPlot, MultivariateStats, BSON

### load model ###
using BSON: @load
@load "models/bandit.bson" m

##### lets do some evaluation ####

test_as = zeros(T, 0)
for i = 1:100
    L, ys, rews, as, ps, hs, states = run_episode(m, ps = [0.20; 0.80], hidden = true)
    global test_as = [test_as as]
end

### plot 'suboptimal pulls' ###
figure(figsize = (6, 4))
imshow(1 .- test_as', cmap = "Greys", aspect = "auto")
xlabel("trial", fontsize = 12)
ylabel("episode", fontsize = 12)
savefig("figs/bandit_suboptimal_pulls.png", bbox_inches = "tight")
close()

### run for different bandit tasks ###
bandits = [[0.99; 0.01], [0.75; 0.25], [0.5;0.5], [0.25; 0.75], [0.01; 0.99]]
all_hs = zeros(Nhidden, 0)
for ps = bandits
    println(ps)
    L, ys, rews, as, ps, hs = run_episode(m, ps = ps, hidden = true)
    global all_hs = [all_hs hs]
    println(mean(rews))
end

### fit PCA across hidden states and reshape to episodes ###
M = MultivariateStats.fit(PCA, all_hs; maxoutdim=3)
Z = transform(M, all_hs)
Z = (Z .- minimum(Z, dims = 2)) ./ (maximum(Z, dims = 2) - minimum(Z, dims = 2))
zs = [Z[:, (i-1)*T+1:i*T] for i = 1:length(bandits)]

### plot figure ###
figure(figsize = (10, 2.))
axs = 150 .+ (1:5)
for i = 1:length(zs)
    subplot(axs[i])
    scatter(zs[i][1,:], zs[i][2,:], c = 1:T, cmap = "coolwarm", s = 5)
    xticks([])
    yticks([])
    xlim(-0.05, 1.05 )
    ylim(-0.05, 1.05 )
    xlabel("PC 1")
    if i == 1
        ylabel("PC 2")
    end
    title(L"$p_L = $"*string(bandits[i][1]))
end
show()
savefig("figs/bandit_dynamics_pca.png", bbox_inches = "tight")
close()
