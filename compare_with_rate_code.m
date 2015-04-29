% simulation time
T = 50;
% step size
stepsize = 0.1; % 0.1 msecs
% number of time points
N = (T/stepsize)+1;
% time input
t = 0:stepsize:T;

% number of neurons
K = 20;
% regularization weight
mu = 1e-8;

% output weight
G = ones(K,1);
G = 0.001 * G;

% generate random walk input
x = abs(cumsum(randn(N, 1)));
x = x * 0.001;
x = smooth(x, floor(N/20));
% derivative
xp = [0; (x(2:N) - x(1:(N-1))) / stepsize];

% simulate network
[V, spike_train, firing_rate, xhat] = simulate_network(N, stepsize, x, xp, K, mu, G);

%%
% simulate rate code network (i.e., poisson spiking)
% we simply replace each neuron with a poisson spiking neuron with the
% same instantaneous firing rates.
% QUESTION: what should we use as firing rate? Should we use spike_train or
% the firing_rate? firing_rate is spike_train convolved with an exponential
% kernel (hence, effect of a spike decays over time. we can see positive
% firing rates in timesteps where the neuron did not spike.)
% Using firing_rate makes the rate code network fire more than the spiking
% network, which leads to larger errors in prediction. If we use the
% spike_train as rate input to poisson network, we get a firing pattern
% really close to the spiking network and predictions become closer as
% well.
poisson_spike_train = poissrnd(spike_train);

% calculate prediction
% NOTE: we are simply using the input/output weights (G) for the spiking
% network. But these may not be the optimal weights for the rate code
% network. One needs to calculate the covariance matrix of the responses
% and the tuning curves to calculate optimal weights via w ~ inv(C)*f', but
% we haven't done that yet.
% C and f' are estimated in tuning_corr_cov script. It is not
% straightforward to take those estimates and use them for calculating the
% optimal weight because f' depends on the stimulus input. Therefore it is
% not clear what f' should be.
poisson_xhat = zeros(N,1);
poisson_xhat(1) = stepsize * G'*poisson_spike_train(:, 1);
for i = 2:N
    dxh = -poisson_xhat(i-1) + G'*poisson_spike_train(:, i);
    poisson_xhat(i) = poisson_xhat(i-1) + (stepsize * dxh);
end

% plot predictions
figure
hold on
plot(t, x)
plot(t, xhat)
plot(t, poisson_xhat)
xlabel('time')
ylabel('prediction')
legend('x', 'xhat-spiking network', 'xhat-rate code network')
%print('fig/spikingnetwork_vs_ratecode', '-dpng')

% plot spike trains
figure
[I,J] = find(spike_train'>0);
scatter(I, J)
axis([0 N+1 0 K+1])
xlabel('timestep')
%print('fig/spikingnetwork_spiketrain', '-dpng')

figure
[I,J] = find(poisson_spike_train'>0);
scatter(I, J)
axis([0 N+1 0 K+1])
xlabel('timestep')
%print('fig/ratecode_spiketrain', '-dpng')


