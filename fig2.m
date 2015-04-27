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
% x = ones(N, 1) * 0.01;
% x = ones(N, 1) * 0.002;
% xp = zeros(N, 1);

% linear ramp input
% x = 0:(0.02/(N-1)):0.02;
% x = x';
% xp = [0; (x(2:N) - x(1:(N-1))) ./ stepsize];
% 
% % random walk input
x = (cumsum(randn(N, 1)));
x = x * 0.0005;
x = smooth(x, 200);
% derivative
xp = [0; (x(2:N) - x(1:(N-1))) / stepsize];

%% Multiple neuron case

% number of neurons
K = 50;

% output weight
G = [ones(K/2,1); -ones(K/2,1)];
G = 0.002 * G;
% G = randn(K,1) * 0.0001;

% recurrent connection weights (these are learned)
w = (0.0001^2) * ones(K,K,N);

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
rate = 0.05;

%w = repmat(G*G' + mu*eye(K), 1, 1, N);

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
    
    % update weights
    do = -obar(:,i-1) + o(:,i-1);
    obar(:,i) = obar(:,i-1) + (stepsize * do);
    w(:,:,i) = w(:,:,i-1) + rate*V(:,i)*obar(:,i)';
    
    wdiff(i) = sum(sum((w(:,:,i) - w_opt).^2)) / sum(sum(w_opt.^2));
    
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

figure(2)
plot(t, obar)
