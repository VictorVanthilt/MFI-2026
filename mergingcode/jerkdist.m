function x = jerkdist(j,tp,lambda,t)

if t <= tp
    x = j*(t^3) / 6; % Calculate distance based on jerk and time
elseif tp<t && t<=2*tp
    x = (j*tp^3)/6 - (j*(t - tp)*(t^2 - 5*t*tp + tp^2))/6;
elseif (2*tp)<t && t<=(2*tp+lambda)
    x = j*tp^3 + j*tp^2*(t - 2*tp);
elseif (2*tp+lambda)<t && t<=(3*tp+lambda)
    x = j*tp^3 + (j*(lambda^3 - 3*lambda^2*t + 6*lambda^2*tp + 3*lambda*t^2 - 12*lambda*t*tp + 6*lambda*tp^2 - t^3 + 6*t^2*tp - 6*t*tp^2 - 4*tp^3))/6 + j*lambda*tp^2;
elseif (3*tp+lambda)<t && t<=(4*tp+lambda)
    x = j*tp^3 - (j*(lambda^3 - 3*lambda^2*t + 12*lambda^2*tp + 3*lambda*t^2 - 24*lambda*t*tp + 48*lambda*tp^2 - t^3 + 12*t^2*tp - 48*t*tp^2 + 63*tp^3))/6 + (j*(6*lambda*tp^2 + 6*lambda^2*tp + 3*lambda*(lambda + 3*tp)^2 - 3*lambda^2*(lambda + 3*tp) + lambda^3 + 6*tp*(lambda + 3*tp)^2 - 6*tp^2*(lambda + 3*tp) - 4*tp^3 - (lambda + 3*tp)^3 - 12*lambda*tp*(lambda + 3*tp)))/6  + j*lambda*tp^2;
end

end