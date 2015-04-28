%% Multiple neuron learning with constant input
% This script implements learning for a network of multiple neurons 
% with constant input in the model proposed by 
%   Bourdoukan R, Barrett DGT, Machens CK, Deneve S (2012), 
%   Learning optimal spike based representations,
%   Advances in Neural Information Processing Systems (NIPS) 25.
% Note that the network implementation is exactly the same with fig2.m
% (also, the multiple neuron case is just a higher dimensional version of
% the single neuron case implemented in fig1.m). 
% This script is intended for generating the multiple neuron plots in 
% Figure 1 in the paper.
%
% 27 April 2015
% Goker Erdogan
clear 
close all
%% Input signal and shared parameters

% simulation time (in mseconds)
T = 200;
% step size
stepsize = 0.1; % 0.1 msecs
% number of time points
N = (T/stepsize)+1;
% time input
t = 0:stepsize:T;

% constant input
x = ones(N, 1) * 0.013;
xp = zeros(N, 1);

%% Multiple neuron learning with constant input

% number of neurons
K = 10;

% output weight
G = [ones(K/2,1); ones(K/2,1)];
G = 0.0003 * G;

% recurrent connection weights (these are learned)
w = (0.001^2) * ones(K,K,N);

% regularization weight 
mu = 1e-8;

% spike train
o = zeros(K,N);

% instantaneous firing rate
obar = zeros(K,N);

% v(t)
V = zeros(K,N);

% xhat(t) (prediction)
xhat = zeros(N,1);

% learning rate
rate = 0.005;

% distance of w to optimal weight matrix
wdiff = zeros(N,1);
w_opt = G*G' + mu*eye(K);
wdiff(1) = sum(sum((w(:,:,1) - w_opt).^2)) / sum(sum(w_opt.^2));

for i = 2:N
    % we omit the reset term because voltages are reset whenever there is a
    % spike (in the below for loop)
    dv = -V(:,i-1) + (G * (x(i-1) + xp(i-1)));
    V(:,i) = V(:,i-1) + (stepsize * dv);
    
    % we use the correct weights for calculating thresholds
    threshold = (G*G' + mu*eye(K));
    
    for dt = 1:K
        % any neuron above threshold?
        diff = V(:,i) - diag(0.5 * (threshold));
        if max(diff) > 0
            % find all neurons above threshold
            maxk = find(diff==max(diff));
            % pick one randomly and fire
            k = maxk(randi(numel(maxk)));
            o(k, i) = (1 / stepsize);
            
            % update prediction (NOTE we use the learned recurrent weights)
            reset_amount = w(:,:,i-1);
            V(:,i) = V(:,i) - (reset_amount(:,k));
            
        else % if no neuron fired, advance to the next timestep
            break;
        end
    end
    
    % update instantaneous firing rate
    do = -obar(:,i-1) + o(:,i-1);
    obar(:,i) = obar(:,i-1) + (stepsize * do);
    % update weights
    w(:,:,i) = w(:,:,i-1) + rate*V(:,i)*obar(:,i)';
    
    % calculate difference from optimal weight matrix
    wdiff(i) = sum(sum((w(:,:,i) - w_opt).^2)) / sum(sum(w_opt.^2));
    
    % update prediction
    dxh = -xhat(i-1) + G'*o(:, i);
    xhat(i) = xhat(i-1) + (stepsize * dxh);
end


figure(1)
subplot(2,2,1)
semilogy(t, x);
hold on
semilogy(t, xhat);
legend('x', 'xhat')
title('x, xhat')

subplot(2,2,2)
plot(t, V(1,:));
title('V');
 
subplot(2, 2, 3)
[I,J] = find(o'>0);
scatter(I, J)
axis([0 N+1 0 K+1])
title('o')

subplot(2,2,4)
plot(t, wdiff)
title('||w-w*||')
print('fig/fig1_multi', '-dpng')

% save figures
figure
hold on
plot(t,x)
plot(t, xhat)
xlabel('time')
ylabel('output')
legend('x', 'xhat')
print('fig/fig1_multi_xxhat', '-dpng')

figure
hold on
plot(t,wdiff)
xlabel('time')
ylabel('||w-w*||/||w*||')
print('fig/fig1_multi_weight', '-dpng')

figure
hold on
[I,J] = find(o'>0);
scatter(I, J)
axis([0 N+1 0 K+1])
xlabel('timestep')
print('fig/fig1_multi_spiketrain', '-dpng')