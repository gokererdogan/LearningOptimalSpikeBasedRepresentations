%% Input signal and shared parameters

% simulation time (in seconds)
T = 2;
% step size
stepsize = 0.01;
% number of time points
N = (T/stepsize)+1;
% time input
t = 0:stepsize:T;

% a simple step function input
x = zeros(N, 1);
x(30:50) = 0.1;
x(70:85) = 0.1;
xp = zeros(N, 1);
xp(30) = 1;
xp(51) = -1;
xp(70) = 1;
xp(86) = -1;

% sine wave input
x = sin(t)*0.1;
x = x';
xp = cos(t)*0.1;
xp = xp';

% random walk input
x = abs(cumsum(randn(N, 1)));
x = x * 0.005;
x = smooth(x, 10);
% derivative
xp = [0; (x(2:N) - x(1:(N-1))) / stepsize];
%% Single neuron case

% current membrane voltage
v = 0;
% output weight
w = 0.1;
% spike train
o = zeros(N,1);
% v(t)
V = zeros(N,1);
% xhat(t) (prediction)
xhat = zeros(N,1);

V2 = zeros(N, 1);
o2 = zeros(N, 1);
xhat2 = zeros(N,1);
xhat3 = zeros(N,1);

xh = 0;
dxh = 0;

for i = 2:N
    % membrane voltage update equation (Eqn. 4)
    dv = -v + w*(x(i-1) + xp(i-1)) - w^2*o(i-1);
    v = v + (stepsize * dv);
    V(i) = v;
    
    % below is the solution of above update equation (Eqn. 4)
    % v = exp(-t) * sum(- stepsize * w * exp(1:t) * (w*o(1:t) - x(1:t) - xp(1:t)));
    
    % fire spike if V > T
    % I found out I need to divide by stepsize (or multiply the 
    % threshold with stepsize) to get the neuron respond fast enough to
    % prediction error. Otherwise the neuron starts firing way after the
    % prediction lags behind the actual signal we want to follow.
    % I don't really have any theoretical justifications of this; I guess
    % it also comes from discretization of the differential equation for
    % spiking rule
    if (v/stepsize) > w^2/2
        o(i) = 1;
    end
    
    % alternative membrane voltage equation
    % gives pretty much the same results with the above voltage update
    v2 = w*(x(i-1) - xhat2(i-1)); 
    V2(i) = v2;
    
    if (v2/stepsize) > w^2/2
        o2(i) = 1;
    end
    
    % calculate prediction
    % Wolfram Alpha can solve the differential equation in Eqn.1 
    % http://www.wolframalpha.com/input/?i=dx%28t%29%2Fdt+%3D+-x%28t%29+%2B+w*o%28t%29
    % The solution is
    %   x(t) = c*exp(-t) + exp(-t) * integral_{1 to T} exp(z)*w*o(z) dz
    % Since o(z) is a spike train
    %   integral_{1 to T} exp(z)*w*o(z) dz = sum_{z in spike_times} w*exp(z)
    % Then
    % x(t) = exp(-t) * (c + sum_{spike times} w*exp(z))
    % What about the initial condition? What should we set c? If we assume
    % x(0) = 0, then c = 0. Then x(t) becomes
    % x(t) = exp(-t) * (sum_{spike times} w*exp(z))
    
    % prediction from the first way of calculating voltage
    xhat(i) = exp(-t(i)) * (stepsize * (w * (exp(t(1:i)) * o(1:i))));
    % prediction from the second way of calculating voltage
    xhat2(i) = exp(-t(i)) * (stepsize * (w * (exp(t(1:i)) * o2(1:i))));
    
    % yet another way to calculate the prediction
    % discretize the differential equation xhat' = -xhat + w*o
    dxh = -xh + w*o2(i);
    xh = xh + (stepsize * dxh);
    xhat3(i) = xh;
end

hold on
plot(t, V)
plot(t, x)
plot(t, xhat)
scatter(t, o)
legend('V', 'x', 'xhat', 'o')

figure
hold on
plot(t, V2)
plot(t, x)
plot(t, xhat2)
plot(t, xhat3)
scatter(t, o2)
legend('V2', 'x', 'xhat2', 'xhat3', 'o2')

%% Homogeneous network of neurons
% TODO: add another population of neurons with negative weights!

% number of neurons
K = 50;
% regularization weight
mu = 1e-3;

% current membrane voltage
v = zeros(K,1);
% output weight
w = 0.1*ones(K,1);
% spike train
o = zeros(K,N);
o2 = zeros(K,N);
% v(t)
V = zeros(K,N);
V2 = zeros(K,N);
% xhat(t) (prediction)
xhat = zeros(K,1);
xhat2 = zeros(K,1);
xhat3 = zeros(K,1);
xh = 0;

% THINK: why don't we get the same results from different ways of
% calculating voltage (V and V2)?

for i = 2:N
    % update voltage neuron by neuron (otherwise, neurons seems to fire
    % synchronously)
    prev_o = o(:,i-1);
    % go in random order over neurons
    % TODO: we can get the same effect (random firing) with adding some
    % noise to the voltage updates. this is what's done in the paper.
    n_order = randperm(K);
    xhatc = xhat3(i-1);
    for j = 1:K
        k = n_order(j);
        % membrane voltage update equation
        dv = -v + (w * (x(i-1) + xp(i-1))) - ((w*w' + mu*eye(K))*prev_o);
        % THINK: should I divide dv by K or not?
        v = v + (stepsize * dv);
        V(:,i) = v;
        
        % fire spike if V_i > T_i
        if (v(k) / stepsize) > (w(k)^2/2 + (mu/2))
            o(k,i) = 1;
            % neuron n spiked, other neurons should know about this.
            prev_o(k) = prev_o(k) + 1;
        end
        
        % second way of calculating voltage
        V2(:,i) = w*(x(i) - xhatc) - mu*mean(o2(:,1:i), 2);
        
        % fire spike if V_i > T_i
        if (V2(k,i) / stepsize) > (w(k)^2/2 + (mu/2))
            o2(k,i) = 1;
            % neuron n spiked, other neurons should know about this.
            dxhatc = -xhatc + w(k);
            xhatc = xhatc + (stepsize * dxhatc);
        end
        
    end
    
    % calculate prediction 
    xhat(i) = exp(-t(i)) * (stepsize * (w' * o(:, 1:i) * exp(t(1:i))'));
    % yet another way to calculate the prediction
    % discretize the differential equation xhat' = -xhat + w*o
    dxh = -xh + w'*o(:, i);
    xh = xh + (stepsize * dxh);
    xhat2(i) = xh;
    
    xhat3(i) = exp(-t(i)) * (stepsize * (w' * o2(:, 1:i) * exp(t(1:i))'));
end

hold on
plot(t, x)
plot(t, xhat)
plot(t, xhat2)
plot(t, xhat3)
legend('x', 'xhat', 'xhat2', 'xhat3')

% spike train
figure
[I,J] = find(o'>0);
scatter(I, J)
axis([0 N+1 0 K+1])
xlabel('time')
ylabel('neuron')

% spike train
figure
[I,J] = find(o2'>0);
scatter(I, J)
axis([0 N+1 0 K+1])
xlabel('time')
ylabel('neuron')

