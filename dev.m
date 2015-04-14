%% Input signal and shared parameters

% number of timesteps
T = 100;
ts = linspace(0, 1, T);

% a simple step function input
x = zeros(T, 1);
x(30:50) = 0.1;
x(70:85) = 0.1;
xp = zeros(T, 1);
xp(30) = 1;
xp(51) = -1;
xp(70) = 1;
xp(86) = -1;

% sine wave input
x = sin(ts)*0.3;
x = x';
xp = cos(ts)*0.3;
xp = xp';

%% Single neuron case

% current membrane voltage
v = 0;
% output weight
w = 0.1;
% spike train
o = zeros(T,1);
% v(t)
V = zeros(T,1);
% xhat(t) (prediction)
xhat = zeros(T,1);


V2 = zeros(T, 1);
o2 = zeros(T, 1);
xhat2 = zeros(T,1);

for t = 2:T
    % membrane voltage update equation
    % There is something wrong with this update (Eqn. 4 in the paper)
    % With this update, the neuron spikes continuously. Imagine that
    % c = x + x' = 0.3, w = 0.1, o can take values 0 and 1; then, dv is 
    % always positive.
    dv = -v + w*(x(t-1) + xp(t-1)) - w^2*o(t-1);
    v = v + dv;
    
    % below is the solution of above update equation (Eqn. 4)
    % v = exp(-t) * sum(-w * exp(1:t) * (w*o(1:t) - x(1:t) - xp(1:t)));
    
    V(t) = v;
    
    % fire spike if V > T
    if v > w^2/2
        o(t) = 1;
    end
    
    % alternative membrane voltage equation
    % this seems to work better
    v2 = w*(x(t) - xhat2(t-1)); % current prediction is previous timestep's prediction
    V2(t) = v2;
    
    if v2 > w^2/2
        o2(t) = 1;
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
    xhat(t) = exp(-t) * (w * (exp(1:t) * o(1:t)));
    
    xhat2(t) = exp(-t) * (w * (exp(1:t) * o2(1:t)));
end

hold on
plot(ts, V)
plot(ts, x)
plot(ts, xhat)
scatter(ts, o)
legend('V', 'x', 'xhat', 'o')

figure
hold on
plot(ts, V2)
plot(ts, x)
plot(ts, xhat2)
scatter(ts, o2)
legend('V2', 'x', 'xhat2', 'o2')

%% Homogeneous network of neurons
% number of neurons
N = 20;
% regularization weight
mu = 1e-5;

% current membrane voltage
v = zeros(N,1);
% output weight
w = 0.5*ones(N,1);
% spike train
o = zeros(N,T);
% v(t)
V = zeros(N,T);
% xhat(t) (prediction)
xh = zeros(T,1);
xhat = zeros(T,1);

for t = 2:T
    % update voltage neuron by neuron (otherwise, neurons seems to fire
    % synchronously)
    prev_o = o(:,t-1);
    % go in random order over neurons
    n_order = randperm(N);
    for i = 1:N
        n = n_order(i);
        % membrane voltage update equation
        dv = -v + w*(x(t-1) + xp(t-1)) - (w*w' + mu*eye(N))*prev_o;
        v(n) = v(n) + dv(n);
        V(n,t) = v(n);
        
        % fire spike if V_i > T_i
        if v(n) > (w(n)^2/2 + (mu/2))
            o(n,t) = 1;
            % neuron n spiked, other neurons should know about this.
            prev_o(n) = prev_o(n) + 1;
        end
        
        % o(v > diag(w*w')/2 + (mu/2), t) = 1;
    end
    
    % calculate prediction 
    % Use the (probably) wrong way
    xh(t) = (w'*o(:,t) + xh(t-1))/2;
    % Use the exact solution for xhat
    % !!! I am not sure if this is the right solution for the multiple
    % neuron case though. !!!
    xhat(t) = exp(-t) * (1 + (w' * o(:, 1:t) * (exp(1:t)')));
end

hold on
% plot(ts, V)
plot(ts, x)
plot(ts, xh)
plot(ts, xhat)
legend('x', 'xh', 'xhat')

% spike train
figure
[I,J] = find(o'>0);
scatter(I, J)
axis([0 T+1 0 N+1])
xlabel('time')
ylabel('neuron')

