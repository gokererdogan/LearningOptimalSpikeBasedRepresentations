% This script just tests if we discretized the differential equation in the
% paper correctly. The analytical and the discretized solutions should be
% the same. This was mainly an exercise to learn how to discretize
% differential equations.
%
% 18 April 2015

T = 10;
stepsize = 0.01;
n = (T/stepsize)+1;
t = 0:stepsize:T;

% the discrete approx. and the analytical solutions match but changing the
% stepsize changes the values you get. that is not right. why does that
% happen? I suspect it is because of not knowing how to discretize a delta
% function. should I keep its value at 1, even if I change the stepsize.
% If I scale the value of delta function using stepsize (as seen below), I
% can get rid of that problem.
spike_times = 1:T;
o = zeros(n, 1);
o(rand(1, n)<0.2) = (1/stepsize);
% o((spike_times * ((n-1)/T))+1) = (1/stepsize);

xhat = zeros(n, 1);
xh = zeros(n, 1);
dx = 0;
w = 0.1;

for i = 2:n
    dx = -xh(i-1) + w * o(i-1);
    xh(i) = xh(i-1) + stepsize*dx;
    
    % analytical solution for constant input of 1, o(t) = 1
    % xhat(i) = w - (w * exp(-t(i)));
    
    % analytical solution for the general case
    % x(t) = exp(-t) * (integral_{1 to t} w*exp(z)*o(z)*dz)
    % Approximate the integral with sum (note the stepsize)
    % x(t) = exp(-t) * stepsize * (sum_{spike times} w*exp(z))
    xhat(i) = exp(-t(i)) * (w * exp(t(1:i)) * o(1:i)) * stepsize;
    % xhat(i) = (exp(-t(i)) * (w * exp(t(1:i)) * o(1:i))) + (w * (exp(1) - 1) * exp(-t(i)));
end


hold on
plot(t, xh)
plot(t, xhat)
legend('discrete approx.', 'analytical')