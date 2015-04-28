%% Single neuron learning
% This script implements learning for a single neuron with constant input
% in the model proposed by 
%   Bourdoukan R, Barrett DGT, Machens CK, Deneve S (2012), 
%   Learning optimal spike based representations,
%   Advances in Neural Information Processing Systems (NIPS) 25.
%
% This script is intended for generating the single neuron plots in Figure
% 1 in the paper.
% 27 April 2015
% Goker Erdogan
clear
close all

%% Input signal and shared parameters

% simulation time (in mseconds)
T = 20;
% step size
stepsize = 0.1; % 0.1 msecs
% number of time points
N = (T/stepsize)+1;
% time input
t = 0:stepsize:T;

% constant input
x = ones(N, 1) * 0.015;
xp = zeros(N, 1);

%% Single neuron learning

% fixed output weight
G = 0.01;

% autapse weight
initial_w = 0.000001;
w = ones(N,1) * initial_w;

% spike train
o = zeros(N,1);

% v(t)
V = zeros(N,1);

% xhat(t) (prediction)
xhat = zeros(N,1);

% instantaneous firing rate
obar = zeros(N,1);

% learning rate
rate = 0.05;

for i = 2:N
    % membrane voltage update equation (Eqn. 4)
    dv = -V(i-1) + G*(x(i-1) + xp(i-1)) - w(i-1)*o(i-1);
    V(i) = V(i-1) + (stepsize * dv);
    
    % if above threshold, spike
    if V(i) > G^2/2
        % note we need to scale the delta function with stepsize
        o(i) = (1 / stepsize);
    end
    
    % update instantaneous firing rate
    do = -obar(i-1) + o(i-1);
    obar(i) = obar(i-1) + (stepsize * do);
    
    % update autapse weight
    w(i) = w(i-1) + rate*V(i)*obar(i);
    
    % update prediction
    dxh = -xhat(i-1) + G*o(i);
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
plot(t, V);
title('V');
 
subplot(2, 2, 3)
scatter(t, o)
title('o')

subplot(2,2,4)
plot(t, w)
title('w')

% save figures
figure
hold on
plot(t,x)
plot(t, xhat)
xlabel('time')
ylabel('output')
legend('x', 'xhat')
print('fig/fig1_xxhat', '-dpng')

figure
hold on
plot(t,V)
xlabel('time')
ylabel('voltage')
print('fig/fig1_voltage', '-dpng')

figure
hold on
plot(t,w)
xlabel('time')
ylabel('weight')
print('fig/fig1_weight', '-dpng')

figure
hold on
scatter(t,o*stepsize)
axis([t(1) t(numel(t)) -0.1 1.1])
xlabel('time')
print('fig/fig1_spiketrain', '-dpng')