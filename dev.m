n = 100;
ts = linspace(0, 1, n);

x = zeros(n, 1);
x(30:50) = 0.1;
x(70:85) = 0.1;
xp = zeros(n, 1);
xp(30) = 1;
xp(51) = -1;
xp(70) = 1;
xp(86) = -1;

x = sin(ts)./4;
xp = cos(ts)./4;

v = 0;
w = 0.1;
o = zeros(n,1);
V = zeros(n,1);
xh = zeros(n,1);

for t = 2:n
    dv = -v + w*(x(t-1) + xp(t-1)) - w^2*o(t-1);
    v = v + dv;
    V(t) = v;
    
    if v > w^2/2
        o(t) = 1;
    end
    
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
