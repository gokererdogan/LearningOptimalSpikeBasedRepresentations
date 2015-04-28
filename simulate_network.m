function [V, spike_train, firing_rate, xhat] = simulate_network(N, stepsize, x, xp, K, mu, G)
% simulate_network(N, stepsize, x, xp, K, mu, G)
% This function simulates the recurrent neural network model given in
%   Bourdoukan R, Barrett DGT, Machens CK, Deneve S (2012), 
%   Learning optimal spike based representations,
%   Advances in Neural Information Processing Systems (NIPS) 25.
%
% This function does *NOT* implement learning. See fig1.m, fig2.m for
% implementations of learning for the single and multiple neuron case
%
% N: number of timesteps in simulation
% stepsize: step size (in seconds)
% x,xp: input and its derivative
% K: number of neurons in the total population
% mu: regularization weight
% G: input/output (decoding) weights. Optional. If not provided, G is set
%   randomly
%
% 27 April 2015
% Goker Erdogan


% membrane voltage v(t)
V = zeros(K,N);

% spike train
spike_train = zeros(K,N);

% xhat(t) (prediction)
xhat = zeros(K,1);

% instantaneous firing rate
firing_rate = zeros(K,N);

if nargin < 6 % G is not provided
    % randomly initialize output weight
    G = randn(K, 1) * 0.1;
end

% voltage reset amounts
reset_amount = G*G' + mu*eye(K);
% neuron firing thresholds
threshold = diag(reset_amount);

for i = 2:N
    % we omit the reset term because voltages are reset whenever there is a
    % spike (in the below for loop)
    dv = -V(:,i-1) + (G * (x(i-1) + xp(i-1)));
    V(:,i) = V(:,i-1) + (stepsize * dv);
    
    % in order to get asynchronou firing, we need to let neurons fire one
    % by one. after a neuron fires, it should reset the voltages of other
    % neurons. that is why we loop multiple times and let multiple neurons
    % fire in the same time step.
    for dt = 1:K % we don't need to loop exactly K times; one can pick another value as well
        % are there neurons above threshold?
        diff = V(:,i) - (0.5 * threshold);
        if max(diff) > 0
            % find all neurons above threshold
            maxk = find(diff==max(diff));
            % pick one randomly and fire
            k = maxk(randi(numel(maxk)));
            spike_train(k, i) = (1 / stepsize);
            
            % update prediction
            V(:,i) = V(:,i) - (reset_amount(:,k));
            
        else % if no neuron fired, advance to the next timestep
            break;
        end
    end
    
    % update instantaneous firing rate
    do = -firing_rate(:,i-1) + spike_train(:,i-1);
    firing_rate(:,i) = firing_rate(:,i-1) + (stepsize * do);
    
    % update prediction
    dxh = -xhat(i-1) + G'*spike_train(:, i);
    xhat(i) = xhat(i-1) + (stepsize * dxh);
end

end