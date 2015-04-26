function [V, spike_train, xhat] = simulate_heterogeneous_network(T, stepsize, x, K, mu, w)
% K: number of neurons in the total population
% mu: regularization weight
% T: simulation time (in seconds)
% stepsize: step size (in seconds)

% number of time points
N = (T/stepsize)+1;

% time input
t = 0:stepsize:T;

% membrane voltage v(t)
V = zeros(K,N);

% spike train
spike_train = zeros(K,N);

% xhat(t) (prediction)
xhat = zeros(K,1);

if nargin < 6 % w is not provided
    % randomly initialize output weight
    w = randn(K, 1) * 0.1;
end

for i = 2:N
    % current xhat. we need this for the second voltage update method.
    xhatc = xhat(i-1);
    
    % if we update all neurons at the same time and let all the neurons
    % that exceed threshold fire, we get synchronous firing. If a neuron
    % fires, it should reset its own voltage and others because now the 
    % network makes a better prediction.
    % in other words, to get asynchronous spiking, we need to let only a 
    % single neuron fire at one time, and update the voltages after each
    % spike.
    % in the paper, they also mention adding noise to voltages will make
    % them fire asynchronously but we still need to fire one spike at a
    % time. Our simulations so far show that we don't need to add noise.
    
    % try firing K times (K is arbitrary here, but it makes sense to make 
    % it as large as the number of neurons)
    for dt = 1:K 
        
        % Voltage Update (below Eqn. 14 in the paper)
        % the second term is mu*o_bar. o_bar is given below eqn. 6 in the
        % paper
        V(:,i) = w*(x(i) - xhatc) - (mu * exp(-t(i)) * spike_train(:,1:i) * exp(t(1:i))');
        
        % NOTE: this does not prevent a neuron from firing multiple times
        % in one step (of course that is not possible, but it will try to
        % fire). this does not seem to be a problem though.
        
        % fire a single neuron, the one with maximum difference
        diff = (V(:,i) / stepsize) - diag(0.5 * (w*w' + (mu*eye(K))));
        % pick all the neurons with maximum voltage
        if max(diff) > 0
            maxk = find(diff==max(diff));
            % pick one randomly and fire
            k = maxk(randi(numel(maxk)));
            spike_train(k, i) = 1;
            % update prediction
            dxhatc = -xhatc + w(k);
            xhatc = xhatc + (stepsize * dxhatc);
        end
        
    end
    
    % calculate prediction 
    xhat(i) = exp(-t(i)) * (stepsize * (w' * spike_train(:, 1:i) * exp(t(1:i))'));
    
end
end