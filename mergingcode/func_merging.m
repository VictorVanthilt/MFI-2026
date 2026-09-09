function [moveP2,time_P12] = func_merging(p1,p2,p3,moveP1)
%%
%%% moveP1.Flag_standstill=Flag_standstill;
%%% moveP1.jerk;
%%% moveP1.tp=tp;
%%% moveP1.lambda=lambda;
%%% moveP1.vp=vp;
%%% moveP1.time_remi=time_remi;
%%% moveP1.i=axis;
%%% moveP1.ti=ti;
%%% moveP1.xi=xi;
%%% moveP1.Flag_dec=Flag_dec;
%----------------------------------

% parameters in matrix
par(1,1) = % a_max 
par(1,2) = % v_max 
par(1,3) = % t_sway
 
par(2,1) = 0.25;% a_max 
par(2,2) = 1;% v_max 
par(2,3) = 5;% t_sway



% We are at standstill at P1. We may perform one of two merge moves.
% We check below if the path between p1 and p2 is vert/horz, or
% if the path between p2 and p3 is vert/horz
if moveP1.Flag_standstill
    if abs(p2(1)-p1(1)) == 0
        i = 2; % pure vertical movement
        flag_merge1a = 1;
    elseif abs(p2(2)-p1(2)) == 0
        i = 1; % pure horizontal movement
        flag_merge1a = 1;
    elseif abs(p3(1)-p2(1)) == 0
        i = 2;
        j = 1;
        flag_merge1b = 1;
    elseif abs(p3(2)-p2(2)) == 0
        i = 1;
        j = 2;
        flag_merge1b = 1;
    end
    
    %% flag_merge1a == 1
    % The first trajectory only consists of a move in the bridge direction 
    % (resp. trolley direction) and the second trajectory consists of a 
    % move in both directions. 
    % In this case the movements in the bridge direction (resp. trolley direction)
    % is joined to one move where no intermediary acceleration 
    % or deceleration takes place. The movement in the 
    % trolley direction (resp. bridge direction) is delayed until 
    % the first intermediary point is reached.
    if flag_merge1a == 1
        distance = abs(p3(i)-p1(i));
        moveP2.i = i;    
        moveP2.Flag_standstill = 0;
        [time,jerk,tp,lambda,vp] = func_n(par(i,1),par(i,2),par(i,1),distance);
        moveP2.jerk = jerk;
        moveP2.vp = vp;
        moveP2.tp = tp;
        moveP2.lambda = lambda; 
        moveP2.xi = abs(p2(i)-p1(i));
        moveP2.ti = jerkdistinv(moveP2.jerk,tp,moveP2.lambda,moveP2.xi);
        moveP2.time_remi = time - moveP2.ti;
        if moveP2.ti > (moveP2.lambda+2*moveP2.tp)
            moveP2.Flag_dec = 1;
        else
            moveP2.Flag_dec=0;
        end
        time_P12 = moveP2.ti;
    elseif flag_merge1b == 1
    %% flagmerge1b == 1
    % The first trajectory consists of movement in both the trolley 
    % and bridge direction and the second trajectory only consists of 
    % a move in the bridge direction (resp. trolley direction). 
    % In this case the movements of the bridge direction 
    % (resp. trolley direction) are joined to one move as before, 
    % but this move is delayed such that it is guaranteed that the crane 
    % passes the first intermediary point.
        distance = abs(p2(j)-p1(j));
        [time12j,~] = func_n(par(j,1),par(j,2),par(j,3),distance);
        distance = abs(p3(i)-p1(i));
        [time13i,jerk,tp,lambda,vp] = func_n(par(i,1),par(i,2),par(i,3),distance);
        time12i = jerkdistinv(j, tp, lambda, abs(p2(j)-p1(j)));
        moveP2.i = i;
            moveP2.Flag_standstill = 0;
            moveP2.jerk = jerk;
            moveP2.vp = vp;
            moveP2.tp = tp;
            moveP2.lambda = lambda; 
            moveP2.xi = abs(p2(i)-p1(i));
            moveP2.ti = jerkdistinv(moveP2.jerk,tp,moveP2.lambda,moveP2.xi);
            moveP2.time_remi = time13i - moveP2.ti;
            if moveP2.ti > (moveP2.lambda+2*moveP2.tp)
                moveP2.Flag_dec = 1;
            else
                moveP2.Flag_dec=0;
            end
        if time12i >= time12j
            % if time to move along i is longer than time to move along j,
            % then both movements can start and movement at P2 is defined
            % by the movement in i direction
            time_P12 = time12i;
        else
            % if time to move along j is longer than time to move along i,
            % then first movement j starts, then movement i starts later 
            % such that both movements pass through p2 at the same time
            time_P12 = time12j;
        end
    else
        % no merging
        moveP2.Flag_standstill = 0;
        [time12i,~] = func_n(par(1,1),par(1,2),par(1,3),abs(p2(1)-p1(1)));
        [time12j,~] = func_n(par(2,1),par(2,2),par(2,3),abs(p2(2)-p1(2)));
        time_P12 = max(time12i,time12j);
    end
end
%%
if moveP1.Flag_standstill == 0
    i = moveP1.i; % main axis
    j = 3 - i; %% auxiliary axis
    tp = moveP1.tp;

    % Flag_dec can have two values: 0->  No deceleration at P1 / 1-> Deceleration at P1
    distance_P12j = abs(p2(j)-p1(j)); 
    if distance_P12j~=0
        % trolley:
        [time_P12j,~] = func_n(par(j,1),par(j,2),par(j,3),distance_P12j);
    else 
    time_P12j = 0;
    end

    if (moveP1.Flag_dec==0) && (sign(p3(i)-p2(i))==sign(p2(i)-p1(i)))  && (moveP1.time_remi-time_P12j>=2*tp) 
        % 0) no deceleration at P1
        % 1) no change of direction for i axis move
        % 2) remaining time for the axis i move between p1 and p2 should be
        % more than (time to perform j axis move + 2tp /deceleration time)
        moveP2 = moveP1; % keep jerk, tp, vp and i    
        moveP2.Flag_standstill = 0;
        %time_P12i = abs(p2(i)-p1(i))/moveP1.vp;  %%% time needed to cover the i axis motion from p1 to p2
        % moveP2.time_P23i=time_P12i+ (p3(i)-p2(i))/moveP1.vp - (p2(i)-p1(i))/moveP1.vp;
        moveP2.lambda = moveP1.lambda+abs(p3(i)-p2(i))/moveP1.vp; 
        moveP2.xi = moveP1.xi+abs(p2(i)-p1(i));
        moveP2.ti = jerkdistinv(moveP2.jerk,tp,moveP2.lambda,moveP2.xi);
        moveP2.time_remi = moveP2.lambda + 4*tp - moveP2.ti;
        if moveP2.ti > (moveP2.lambda+2*tp)
            moveP2.Flag_dec = 1;
        else
            moveP2.Flag_dec=0;
        end
        time_P12 = moveP2.ti - moveP1.ti;
    else
        moveP2 = moveP1; %values do not matter, we get to a standstill
        moveP2.Flag_standstill=1; % Conditions for merging are not met!
        time_P12= max(moveP1.time_remi, time_P12j);
    end
end