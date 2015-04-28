%% Multiple neuron learning with random input
% This script implements learning for a network of multiple neurons 
% with random input in the model proposed by 
%   Bourdoukan R, Barrett DGT, Machens CK, Deneve S (2012), 
%   Learning optimal spike based representations,
%   Advances in Neural Information Processing Systems (NIPS) 25.
% Note that the network implementation is exactly the same with fig1_multi.m
% Network code is exactly the same, only the input is different.
% This script is intended for generating some of the plots in Figure 2 in
% the paper
%
% 27 April 2015
% Goker Erdogan
clear 
close all
%% Input signal and shared parameters

% simulation time (in mseconds)
T = 800;
% step size
stepsize = 0.1; % 0.1 msecs
% number of time points
N = (T/stepsize)+1;
% time input
t = 0:stepsize:T;

% % random walk input
x = abs(cumsum(randn(N, 1)));
x = x * 0.001;
x = smooth(x, 200);
% derivative
xp = [0; (x(2:N) - x(1:(N-1))) / stepsize];

%% Multiple neuron case

% number of neurons
K = 50;

% output weight
G = [ones(K/2,1); ones(K/2,1)];
G = 0.02 * G;

% recurrent connection weights (these are learned)
w = (0.01^2) * ones(K,K,N);

% regularization weight 
mu = 1e-6;

% spike train
o = zeros(K,N);

% instantaneous firing rate
obar = zeros(K,N);

% v(t)
V = zeros(K,N);

% xhat(t) (prediction)
xhat = zeros(N,1);

% learning rate
rate = 0.01;

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
        diff = V(:,i) - diag(0.5 * (threshold));
        if max(diff) > 0
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
    
    % calculate difference from optimal weights
    wdiff(i) = sum(sum((w(:,:,i) - w_opt).^2)) / sum(sum(w_opt.^2));
    
    % update prediction
    dxh = -xhat(i-1) + G'*o(:, i);
    xhat(i) = xhat(i-1) + (stepsize * dxh);
end


figure(1)
subplot(2,2,1)
plot(t, x);
hold on
plot(t, xhat);
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
print('fig/fig2_multi', '-dpng')

% save figures
figure
hold on
plot(t,x)
plot(t, xhat)
xlabel('time')
ylabel('output')
legend('x', 'xhat')
print('fig/fig2_xxhat', '-dpng')

figure
hold on
plot(t,wdiff)
xlabel('time')
ylabel('||w-w*||/||w*||')
print('fig/fig2_weight', '-dpng')

figure
hold on
[I,J] = find(o'>0);
scatter(I, J)
axis([0 N+1 0 K+1])
xlabel('timestep')
print('fig/fig2_spiketrain', '-dpng')
