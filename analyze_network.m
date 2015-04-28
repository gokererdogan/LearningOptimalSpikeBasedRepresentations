%% Analyze network dynamics
% This script calculates Fano factor, coefficient of variation of 
% interspike intervals and looks at interspike interval distributions. We
% would expect Fano factors and CVs around 1, and an exponential ISI curve
% from a traditional Poisson model (experimental observations support such
% a model as well). In other words, here we investigate whether the
% proposed network carries known dynamical properties of neural networks.
%
% 27 April 2015
% goker erdogan

% simulation time
T = 100;
% step size
stepsize = 0.1; % 0.1 msecs
% number of time points
N = (T/stepsize)+1;
% time input
t = 0:stepsize:T;

% number of neurons
K = 50;
% regularization weight
mu = 1e-8;

%% Simulate network runs and plot figures

runs = 200;
rate_mean = zeros(K, runs);
isi = cell(K, 1);

% output weight
G = ones(K,1);
G = 0.001 * G;

for r = 1:runs
    r
    % generate random walk input
    x = abs(cumsum(randn(N, 1)));
    x = x * 0.001;
    x = smooth(x, floor(N/20));
    % derivative
    xp = [0; (x(2:N) - x(1:(N-1))) / stepsize];

    % simulate network
    [V, spike_train, firing_rate, xhat] = simulate_network(N, stepsize, x, xp, K, mu, G);
    
    % calculate firing rate and interspike intervals
    rate_mean(:, r) = sum(spike_train, 2) ./ T;
    for k = 1:K
        spike_time = find(spike_train(k, :));
        spike_count = numel(spike_time);
        isi{k} = [isi{k} ((spike_time(2:spike_count) - spike_time(1:(spike_count-1))) * stepsize)];
    end
end

% firing rate mean, variance and fano factor
rm = mean(rate_mean, 2);
rv = var(rate_mean, 0, 2);
fano = rv ./ rm;
figure
histogram(fano, 10)
xlabel('Fano factor')
ylabel('# of neurons')
print('fig/fano', '-dpng')

% isi mean, std. dev. and coefficient of variation
isim = cellfun(@mean, isi); 
isis = cellfun(@std, isi); 
cv = isis ./ isim;
figure 
histogram(cv, 10)
xlabel('Coefficient of Variation')
ylabel('# of neurons')
print('fig/isi_cv', '-dpng')

figure
histogram(isi{1})
xlabel('Interspike interval (msecs)')
print('fig/isi_hist', '-dpng')

%% Comments
% simulation results show that CV values are pretty robust to parameter
% settings, but Fano factors are not. It is in fact not easy to get Fano
% factors around 1. You need to tweak the parameters for some time until
% you get them.