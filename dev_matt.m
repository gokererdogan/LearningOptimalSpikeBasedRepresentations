%% Input signal and shared parameters

% simulation time (in seconds)
T = 8;
% step size
stepsize = 10^-3; % 1 msecs
% number of time points
N = (T/stepsize)+1;
% time input
t = 0:stepsize:T;

% a simple step function input
% x = zeros(N, 1);
% x(30:50) = 0.05;
% x(70:85) = 0.05;
% xp = zeros(N, 1);
% xp(30) = 1;
% xp(51) = -1;
% xp(70) = 1;
% xp(86) = -1;

% sine wave input
% x = sin(t)*0.05;
% x = x';
% xp = cos(t)*0.05;
% xp = xp';

% random walk input
%x = abs(cumsum(randn(N, 1)));
%x = x * 0.005;
%x = smooth(x, 50);

% constant input (to reproduce fig 1.B)
x = ones(N, 1) * 0.1;

% generalized derivative
xp = [diff(x); 0] / stepsize;

%% Single neuron case

% current membrane voltage
v = 0;
% fixed output weight (gamma)
G = 0.1;
% Threshold
T = G^2/2;
% learned weight (w_i)
initial_w = 0.0001;
w = ones(N, 1) * initial_w;
w2 = ones(N, 1) * initial_w;
% learning rate (tau in the paper -- what should this be??)
a = 0.1;
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
    
    % calculate prediction
    % Wolfram Alpha can solve the differential equation in Eqn.1 
    % http://www.wolframalpha.com/input/?i=dx%28t%29%2Fdt+%3D+-x%28t%29+%2B+w*o%28t%29
    % The solution is
    %   x(t) = c*exp(-t) + exp(-t) * integral_{1 to T} exp(z)*G*o(z) dz
    % Since o(z) is a spike train
    %   integral_{1 to T} exp(z)*G*o(z) dz = sum_{z in spike_times} G*exp(z)
    % Then
    % x(t) = exp(-t) * (c + sum_{spike times} G*exp(z))
    % What about the initial condition? What should we set c? If we assume
    % x(0) = 0, then c = 0. Then x(t) becomes
    % x(t) = exp(-t) * (sum_{spike times} G*exp(z))
    
    
    % prediction from the first way of calculating voltage
    xhat(i) = exp(-t(i)) * (stepsize * (G * (exp(t(1:i)) * o(1:i))));
    % prediction from the second way of calculating voltage
    xhat2(i) = exp(-t(i)) * (stepsize * (G * (exp(t(1:i)) * o2(1:i))));
    
    % yet another way to calculate the prediction
    % discretize the differential equation xhat' = -xhat + G*o
    dxh = -xh + G*o2(i);
    xh = xh + (stepsize * dxh);
    xhat3(i) = xh;
    
    % membrane voltage update equation (Eqn. 4)
    c = x(i-1) + xp(i-1);
    dv = -v + G*c - w(i-1)*o(i-1);
    v = v + (stepsize * dv);
    V(i) = v;
    
    % below is the solution of above update equation (Eqn. 4)
    % v = exp(-t) * sum(- stepsize * G * exp(1:t) * (G*o(1:t) - x(1:t) - xp(1:t)));
    
    % fire spike if V > T
    % I found out I need to divide by stepsize (or multiply the 
    % threshold with stepsize) to get the neuron respond fast enough to
    % prediction error. Otherwise the neuron starts firing way after the
    % prediction lags behind the actual signal we want to follow.
    % I don't really have any theoretical justifications of this; I guess
    % it also comes from discretization of the differential equation for
    % spiking rule
    if (v/stepsize) > T
        o(i) = 1;
        w(i) = w(i-1) + w(i-1)*(w(i-1) - 2*T) / (a*(T - G*c - w(i-1)));
    else
        w(i) = w(i-1);
    end
    
    
    % alternative membrane voltage equation
    % gives pretty much the same results with the above voltage update
    v2 = G*(x(i-1) - xhat2(i-1)); 
    V2(i) = v2;
    
    if (v2/stepsize) > T
        o2(i) = 1;
        w2(i) = w2(i-1) + w2(i-1)*(w2(i-1) - 2*T) / (a*(T - G*c - w(i-1)));
    else
        w2(i) = w2(i-1);
    end
    
end


% original plots
%
% figure(1);
% hold on
% plot(t, V)
% plot(t, x)
% plot(t, xhat)
% %scatter(t, o)
% legend('V', 'x', 'xhat')
% print('sing_F1','-dpng')
% 
% figure(2)
% scatter(t, o)
% print('sing_F2','-dpng')
% 
% figure(3)
% hold on
% plot(t, V2)
% plot(t, x)
% plot(t, xhat2)
% plot(t, xhat3)
% %scatter(t, o2)
% legend('V2', 'x', 'xhat2', 'xhat3')
% print('sing_F3','-dpng')
% 
% figure(4)
% scatter(t, o2)
% print('sing_F4','-dpng')


% plot each var separately for visibility
% fig 1: input signal, voltages, output prediction, weights
vars = {x V V2 xhat xhat2 xhat3 w w2};
labels = {['x'] ['V'] ['V2'] ['xhat'] ['xhat2'] ['xhat3'] ['w'] ['w2']};

figure(1)
rows = 4;
cols = 2;
for i = 1:length(vars)
    subplot(rows, cols, i);
    plot(t, vars{i});
    title(labels{i});
end

% fig 2: spikes
figure(2)
vars = {o o2};
labels = {['o'] ['o2']};
for i = 1:length(vars)
    subplot(2, 1, i);
    scatter(t, vars{i}, 'x');
    title(labels{i});
    ylim([-1 2]);
end


%% Homogeneous network of neurons

% number of neurons in the total population
K = 100; % half have positive, half have negative weights

% regularization weight
mu = 1e-6;

% current membrane voltage
v = zeros(K,1);

% output weight
w = 0.1*ones(K,1);
w((K/2+1):K) = -w(1);

% right now, we use two ways to calculate voltages, that is why we have two
% spike trains, voltage and prediction matrices
%
% the first way is using the differential equation for voltage (eqn. 15 in
% the paper). however, it does not work as well as the second way. I don't
% know why exactly, but I think it has something to do with how we update
% the voltages once a neuron fires.
%
% the second way is using the direct definition for voltage (difference
% between input and prediction of the population). This method works pretty
% well.

% THINK: why don't we get the same results from different ways of
% calculating voltage (V and V2)?

% spike train
o = zeros(K,N);
o2 = zeros(K,N);

% v(t)
V = zeros(K,N);
V2 = zeros(K,N);

% xhat(t) (prediction)
xhat = zeros(K,1);
xhat2 = zeros(K,1);

% this expression is used multiple times in updates; it determines how much
% one spike changes the voltage.
reset_amount = (w*w' + mu*eye(K));

for i = 2:N
    % current xhat. we need this for the second voltage update method.
    xhatc = xhat2(i-1);
    
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
        % ----------------------------------------------------------------
        % Voltage Update Method 1
        % membrane voltage update equation
        dv = -v + (w * (x(i-1) + xp(i-1))) - (reset_amount*o(:, i-1));
        v = v + (stepsize * dv);
        V(:,i) = v;
        
        % fire a single neuron, the one with maximum difference
        diff = (V(:,i) / stepsize) - diag(0.5 * (reset_amount));
        % pick all the neurons with maximum voltage
        if max(diff) > 0
            maxk = find(diff==max(diff));
            % pick one randomly and fire
            k = maxk(randi(numel(maxk)));
            o(k, i) = 1;
            % update prediction
            % WARNING: I am not sure if this is the right way to reset!
            v = v - (stepsize * reset_amount(:,k));
        end
        % ----------------------------------------------------------------
        
        % ----------------------------------------------------------------
        % Voltage Update Method 2
        % second way of calculating voltage (below Eqn. 14 in the paper)
        % the second term is mu*o_bar. o_bar is given below eqn. 6 in the
        % paper
        V2(:,i) = w*(x(i) - xhatc) - (mu * exp(-t(i)) * o2(:,1:i) * exp(t(1:i))');
        
        % NOTE: this does not prevent a neuron from firing multiple times
        % in one step (of course that is not possible, but it will try to
        % fire). this does not seem to be a problem though.
        
        % fire a single neuron, the one with maximum difference
        diff = (V2(:,i) / stepsize) - diag(0.5 * (w*w' + (mu*eye(K))));
        % pick all the neurons with maximum voltage
        if max(diff) > 0
            maxk = find(diff==max(diff));
            % pick one randomly and fire
            k = maxk(randi(numel(maxk)));
            o2(k, i) = 1;
            % update prediction
            dxhatc = -xhatc + w(k);
            xhatc = xhatc + (stepsize * dxhatc);
        end
        % ----------------------------------------------------------------
        
    end
    
    % calculate prediction 
    xhat(i) = exp(-t(i)) * (stepsize * (w' * o(:, 1:i) * exp(t(1:i))'));
    xhat2(i) = exp(-t(i)) * (stepsize * (w' * o2(:, 1:i) * exp(t(1:i))'));
    
    % yet another way to calculate the prediction is to 
    % discretize the differential equation xhat' = -xhat + w*o
    % it gives the same prediction with the direct way of calculating
    % predictions
    % dxh = -xh + w'*o(:, i);
    % xh = xh + (stepsize * dxh);
    % xhat2(i) = xh;
    
    
end

figure
hold on
plot(t, x)
plot(t, xhat)
plot(t, xhat2)
legend('x', 'xhat', 'xhat2')
print('pop_F1','-dpng')

% spike train
figure
[I,J] = find(o'>0);
scatter(I, J)
axis([0 N+1 0 K+1])
xlabel('time')
ylabel('neuron')
print('pop_F2','-dpng')

% spike train
figure
[I,J] = find(o2'>0);
scatter(I, J)
axis([0 N+1 0 K+1])
xlabel('time')
ylabel('neuron')
print('pop_F3','-dpng')
