%% check if jerkdist works
clear all
clc
j = 2;
tp = 3;
lambda = 5;
t = linspace(0,lambda+4*tp, 1000);
x = [];
for k = 1:1000
    xnew = jerkdist(j,tp,lambda,t(k));
    x = [x,xnew];
end

plot(t,x)

%% check jerkdistinv

clear all
clc
j = 2;
tp = 3;
lambda = 5;

x = 162;
tiv = jerkdistinv(j, tp, lambda, x);
disp(tiv);

%%
clear all
clc
p0 = [0,0];
p1 = [20;0];
p2 = [40;70];
p3 = [70;70];

a_max=0.25;v_max=1;t_sway=5;
[time,j,tp,lambda,vp] = func_n(a_max,v_max,t_sway,40);

moveP1.Flag_standstill = 0;
moveP1.jerk = j;
moveP1.tp = tp;
moveP1.lambda = lambda;
moveP1.vp = vp;
moveP1.i = 1;
moveP1.ti = jerkdistinv(j,tp,lambda,20);
moveP1.xi = 20;
moveP1.time_remi = time - moveP1.ti;
moveP1.Flag_dec = 0;
[moveP2,time_P12] = func_merging(p1,p2,p3,moveP1)

[time,j,tp,lambda,vp] = func_n(a_max,v_max,t_sway,70)

% TODO:
% 1) merging with v = 0
% 2) sign checking
% 3) case studies