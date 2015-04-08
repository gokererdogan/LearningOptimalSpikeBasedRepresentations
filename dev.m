% number of timesteps
n = 100;
ts = linspace(0, 1, n);

% a simple step function
x = zeros(n, 1);
x(30:50) = 0.1;
x(70:85) = 0.1;
xp = zeros(n, 1);
xp(30) = 1;
xp(51) = -1;
xp(70) = 1;
xp(86) = -1;

% % sine wave input
% x = sin(ts)./4;
% xp = cos(ts)./4;

% current membrane voltage
v = 0;
% output weight
w = 0.1;
% spike train
o = zeros(n,1);
% v(t)
V = zeros(n,1);
% xhat(t) (prediction)
xh = zeros(n,1);


for t = 2:n
    % membrane voltage update equation
    dv = -v + w*(x(t-1) + xp(t-1)) - w^2*o(t-1);
    v = v + dv;
    
    V(t) = v;
    
    % fire spike if V > T
    if v > w^2/2
        o(t) = 1;
    end
    
    % calculate prediction 
    % NOTE: this is probably not the right way to calculate the prediction.
    % Eqn. 1 in paper containt derivative of xhat, I just approximated that
    % here with xhat(t) - xhat(t-1), which is probably not the best idea
    xh(t) = (w*o(t) + xh(t-1))/2;
end

hold on
plot(ts, V)
plot(ts, x)
plot(ts, xh)
scatter(ts, o)
legend('V', 'x', 'xh', 'o')

%%
% change time window size
