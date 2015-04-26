%% Input signal and shared parameters

% simulation time (in mseconds)
T = 50;
% step size
stepsize = 0.01; % 0.1 msecs
% number of time points
N = (T/stepsize)+1;
% time input
t = 0:stepsize:T;

% constant input
x = ones(N, 1) * 0.1;
xp = zeros(N, 1);

%% Single neuron case

% output weight
w = 0.01;

% spike train
o = zeros(N,1);
o2 = zeros(N, 1);

% v(t)
V = zeros(N,1);
V2 = zeros(N, 1);

% xhat(t) (prediction)
xhat = zeros(N,1);
xhat2 = zeros(N,1);

for i = 2:N
    % membrane voltage update equation (Eqn. 4)
    dv = -V(i-1) + w*(x(i-1) + xp(i-1)) - w^2*o(i-1);
    V(i) = V(i-1) + (stepsize * dv);
    
    if V(i) > w^2/2
        o(i) = (1 / stepsize);
    end
    
    % alternative membrane voltage equation
    % gives pretty much the same results with the above voltage update
    V2(i) = w*(x(i-1) - xhat2(i-1)); 
    
    if (V2(i)/stepsize) > w^2/2
        o2(i) = 1;
    end
    
    % prediction from the first way of calculating voltage
    % xhat(i) = exp(-t(i)) * (stepsize * (w * (exp(t(1:i)) * o(1:i))));
    dxh = -xhat(i-1) + w*o(i);
    xhat(i) = xhat(i-1) + (stepsize * dxh);
    % xhat(i) = x(i) - (V(i)/w);
    % prediction from the second way of calculating voltage
    xhat2(i) = exp(-t(i)) * (stepsize * (w * (exp(t(1:i)) * o2(1:i))));
    
end


figure(1)
%plot(t, x)
%plot(t, xhat)
semilogy(t, x);
hold on
semilogy(t, xhat);
axis([0 T 1e-2 1]);
%plot(t, V)
% plot(t, xhat2)
legend('x', 'xhat')

figure(2)
plot(t, V);

figure(3)
subplot(2, 1, 1)
scatter(t, o)
subplot(2, 1, 2)
scatter(t, o2)
