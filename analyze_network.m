% simulation time (in seconds)
T = 0.03;
% step size
stepsize = 0.1*(1/1000); % 0.1 msecs
% number of time points
N = (T/stepsize)+1;
% time input
t = 0:stepsize:T;

% number of neurons
K = 100;
% regularization weight
mu = 1e-6;

%%

runs = 100;
rate_mean = zeros(K, runs);
isi = cell(K, 1);

% simulate with fixed decoding weights
w = randn(K, 1) * 0.1;

for r = 1:runs
    r
    % random walk input
    x = abs(cumsum(randn(N, 1)));
    x = x * 0.0005;
    x = smooth(x, 50);
    dd
    % simulate network
    [V, spike_train, xhat] = simulate_heterogeneous_network(T, stepsize, x, K, mu, w);
    
    rate_mean(:, r) = sum(spike_train, 2) ./ T;
    for k = 1:K
        spike_time = find(spike_train(k, :));
        spike_count = numel(spike_time);s
        isi{k} = [isi{k} (spike_time(2:spike_count) - spike_time(1:(spike_count-1)))];
    end
end

% firing rate mean, variance and fano factor
rm = mean(rate_mean, 2);
rv = var(rate_mean, 0, 2);
fano = rv ./ rm;
figure
histogram(fano)

% isi mean, std. dev. and coefficient of variation
isim = cellfun(@mean, isi); 
isis = cellfun(@std, isi); 
cv = isis ./ isim;
figure 
histogram(cv)

% figure
% hold on
% plot(t, x)
% plot(t, xhat)
% legend('x', 'xhat')
% 
% figure
% [I,J] = find(spike_train'>0);
% scatter(I, J)
% axis([0 N+1 0 K+1])
% xlabel('time')
% ylabel('neuron')

%%
% why simulate multiple runs? 
% simulate a single long trial!