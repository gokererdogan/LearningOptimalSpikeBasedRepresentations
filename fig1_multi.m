clear 
close all
%% Input signal and shared parameters

% simulation time (in mseconds)
T = 50;
% step size
stepsize = 0.1; % 0.1 msecs
% number of time points
N = (T/stepsize)+1;
% time input
t = 0:stepsize:T;

% constant input
x = ones(N, 1) * 0.01;
xp = zeros(N, 1);

% linear ramp input
x = 0:(0.02/(N-1)):0.02;
x = x';
xp = [0; (x(2:N) - x(1:(N-1))) ./ stepsize];

% random walk input
x = abs(cumsum(randn(N, 1)));
x = x * 0.0005;
x = smooth(x, 50);
% derivative
xp = [0; (x(2:N) - x(1:(N-1))) / stepsize];

%% Multiple neuron case

% number of neurons
K = 100;

% output weight
w = [ones(K/2,1); -ones(K/2,1)];
w = 0.0005 * w;
w = randn(K,1) * 0.01;

% regularization weight 
mu = 1e-6;

% spike train
o = zeros(K,N);

% v(t)
V = zeros(K,N);

% xhat(t) (prediction)
xhat = zeros(N,1);

reset_amount = (w*w' + mu*eye(K));

for i = 2:N
%     dv = -V(:,i-1) + (w * (x(i-1) + xp(i-1))) - (reset_amount*o(:, i-1));
%     V(:,i) = V(:,i-1) + (stepsize * dv) + (randn(K,1) * 0);
    
%     diff = (V(:,i)) - diag(0.5 * (reset_amount));
%     if max(diff) > 0
%         maxk = find(diff==max(diff));
%         % pick one randomly and fire
%         k = maxk(randi(numel(maxk)));
%         o(k, i) = (1 / stepsize);
%     end

    dv = -V(:,i-1) + (w * (x(i-1) + xp(i-1)));
    V(:,i) = V(:,i-1) + (stepsize * dv);
    
    for dt = 1:K
        diff = V(:,i) - diag(0.5 * (reset_amount));
        if max(diff) > 0
            maxk = find(diff==max(diff));
            % pick one randomly and fire
            k = maxk(randi(numel(maxk)));
            o(k, i) = (1 / stepsize);
            % update prediction
            V(:,i) = V(:,i) - (reset_amount(:,k));
        end
    end


%     V(:,i) = V(:,i-1);
%     
%     for dt = 1:K
%         diff = V(:,i) - diag(0.5 * (reset_amount));
%         if max(diff) > 0
%             maxk = find(diff==max(diff));
%             % pick one randomly and fire
%             k = maxk(randi(numel(maxk)));
%             o(k, i) = (1 / stepsize);
%         end
%         % update prediction
%         dv = -V(:,i) + (w * (x(i) + xp(i))) - (reset_amount*o(:, i));
%         V(:,i) = V(:,i) + (stepsize * dv / K);
%     end

%     % fire a single neuron, the one with maximum difference
%     diff = (V(:,i)) - diag(0.5 * (reset_amount));
%     % pick all the neurons with maximum voltage
%     while max(diff) > 0
%         maxk = find(diff==max(diff));
%         % pick one randomly and fire
%         k = maxk(randi(numel(maxk)));
%         o(k, i) = (1 / stepsize);
%         % update prediction
%         % WARNING: I am not sure if this is the right way to reset!
%         V(:,i) = V(:,i) - (stepsize * reset_amount(:,k));
% 
%         diff = V(:,i) - diag(0.5 * (reset_amount));
%     end
%     
    % prediction from the first way of calculating voltage
    dxh = -xhat(i-1) + w'*o(:, i);
    xhat(i) = xhat(i-1) + (stepsize * dxh);
end


figure(1)
plot(t, x)
hold on
plot(t, xhat)
%semilogy(t, x);
semilogy(t, xhat);
% axis([0 T 1e-2 1]);
legend('x', 'xhat')

figure(2)
plot(t, V);

figure(3)
[I,J] = find(o'>0);
scatter(I, J)
axis([0 N+1 0 K+1])

