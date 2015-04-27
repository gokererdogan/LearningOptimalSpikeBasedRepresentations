clear
close all

%% Input signal and shared parameters

% simulation time (in mseconds)
T = 20;
% step size
stepsize = 0.01; % 0.1 msecs
% number of time points
N = (T/stepsize)+1;
% time inpute
t = 0:stepsize:T;

% constant input
x = ones(N, 1) * 0.1;
xp = zeros(N, 1);

%% Single neuron case

% fixed output weight
G = 0.05;

% autapse weight
initial_w = 0.000000001;
w = ones(N,1) * initial_w;

% spike train
o = zeros(N,1);

% v(t)
V = zeros(N,1);

% xhat(t) (prediction)
xhat = zeros(N,1);

% instantaeous firing rate
obar = zeros(N,1);
% learning rate
rate = 0.001;

for i = 2:N
    % membrane voltage update equation (Eqn. 4)
    dv = -V(i-1) + G*(x(i-1) + xp(i-1)) - w(i-1)*o(i-1);
    V(i) = V(i-1) + (stepsize * dv);
    
    if V(i) > G^2/2
        o(i) = (1 / stepsize);
    end
    
    %obar(i) = stepsize * exp(-t(i)) .* (o(1:i)' * exp(t(1:i))');
    do = -obar(i-1) + o(i-1);
    obar(i) = obar(i-1) + (stepsize * do);
    w(i) = w(i-1) + rate*V(i)*obar(i);
    
    % alternative membrane voltage equation
    % gives pretty much the same results with the above voltage update
    %V2(i) = G*(x(i-1) - xhat2(i-1)); 
    
    %if (V2(i)/stepsize) > G^2/2
    %   o2(i) = 1;
    %end
    
    % prediction from the first way of calculating voltage
    % xhat(i) = exp(-t(i)) * (stepsize * (G * (exp(t(1:i)) * o(1:i))));
    dxh = -xhat(i-1) + G*o(i);
    xhat(i) = xhat(i-1) + (stepsize * dxh);
    % xhat(i) = x(i) - (V(i)/G);
    % prediction from the second way of calculating voltage
    %xhat2(i) = exp(-t(i)) * (stepsize * (G * (exp(t(1:i)) * o2(1:i))));
    
end

figure(1)
subplot(2,2,1)
semilogy(t, x);
hold on
semilogy(t, xhat);
axis([0 T 1e-2 1]);
legend('x', 'xhat')
title('x, xhat')

subplot(2,2,2)
plot(t, V);
hold on
plot(t, ones(N, 1)*-G^2/2)
title('V');

subplot(2, 2, 3)
scatter(t, o)
%ylim([-1 2])
title('o')

subplot(2,2,4)
plot(t, w)
hold on
plot(t, ones(N, 1)*G^2)
title('w')

% figure(2)
% plot(t, obar)
% title('obar')
