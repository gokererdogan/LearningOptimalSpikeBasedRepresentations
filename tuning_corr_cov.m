%% Tuning curves and response correlations
% This script estimates the tuning curves in a heterogeneous spiking
% network (for the model in Bourdoukan et al. (2012)). We also calculate
% response correlations (and covariance matrix as well). 
%
% 29 April 2015
% goker erdogan

% simulation time
T = 1000;
% step size
stepsize = 0.1; % 0.1 msecs
% number of time points
N = (T/stepsize)+1;
% time input
t = 0:stepsize:T;

% number of neurons
K = 10;
% regularization weight
mu = 1e-8;

%% Simulate network runs
runs = 100;
% output weight
G = linspace(-0.005, 0.005, K);
G = G';

spike_trains = zeros(K, N*runs);
xs = zeros(N*runs, 1);

% set of inputs
ninputs = 40;
inputs = linspace(-0.05, 0.05, ninputs);

for r = 1:runs
    r
%     % generate random walk input
%     x = (cumsum(randn(N, 1)));
%     x = x * 0.001;
%     x = smooth(x, floor(N/20));
%     % derivative
%     xp = [0; (x(2:N) - x(1:(N-1))) / stepsize];
%     

    % generate stepwise constant input
    ci = inputs(randperm(ninputs));
    x = ones(N,1) * ci(ninputs);
    perinputT = round(N / ninputs);
    for i = 1:(ninputs-1)
        x((((i-1)*perinputT)+1):(i*perinputT)) = ci(i);
        xs(((N*(r-1))+1):(N*r), 1) = x;
    end
    xp = zeros(N,1);
    
    % simulate network
    [V, spike_train, firing_rate, xhat] = simulate_network(N, stepsize, x, xp, K, mu, G);
    spike_trains(:, ((N*(r-1))+1):(N*r)) = spike_train;

end

% plot(x)
% hold on
% plot(xhat)

%% tuning curves and f prime
nstep = ninputs; % for stepwise constant input
xstep = (max(xs) - min(xs)) / nstep;
xrange = linspace(min(xs) - xstep/2, max(xs) + xstep/2, nstep+1);
tuning = zeros(K, nstep);

% calculate the mean firing rate for each neuron within a given input
% interval
for s = 1:nstep
    tuning(:, s) = mean(spike_trains(:, xs<xrange(s+1) & xs>=xrange(s)),2);
    % tuning(:, s) = mean(spike_trains(:, xs==inputs(s)),2);
end

plot(tuning')
xlabel('x (input)')
ylabel('firing rate')
print('fig/tuning_curves_1000', '-dpng')


% calculate f' (assuming that tuning curves are linear)
% this is pretty much useless because the tuning curves are not linear and
% f' depends stimulus input
%[I, J] = max(rate_mean');
%fp = I' - diag(rate_mean(:,J-1));

%% response correlations
corrm = corr(spike_trains');
covm = cov(spike_trains');

figure
imagesc(corrm)
colormap('jet')
colorbar
print('fig/response_corr', '-dpng')

figure
imagesc(covm)
colormap('jet')
colorbar
print('fig/response_cov', '-dpng')

