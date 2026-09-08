function [time,j,tp,lambda,vp]=func_n(a_max,v_max,t_sway,distance)
n_max = ceil(v_max / (t_sway * a_max));
n=ceil(sqrt(distance/(2*a_max*t_sway^2)));
if n<n_max
%     j=distance/(2*tp^3)
% ap=j*tp;
% vp=tp*ap;
time=4*n*t_sway;
else 
    n=n_max;
tp=t_sway*n;
ap=v_max/(tp);
vp=ap*tp;
j=ap/tp;
j_bound=min( a_max/tp,v_max/tp^2);
if (j>j_bound)
    error('here1')
end
lambda=(distance-2*j*tp^3)/(j*tp^2);
if lambda<0
    lambda=0;
end
time=lambda+4*tp;
end
