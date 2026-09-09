function t = jerkdistinv(j,tp,lambda,x)
    f = @(t) jerkdist(j,tp,lambda,t)-x;
    t = fsolve(f,1);
end