%%% toy example
%%% Bridge: 0->58
%%% bridge : 
a_max=0.30;v_max=2;t_sway=4;distance=58;
[time1]=func_n(a_max,v_max,t_sway,distance)
%%% trolley: 
a_max=0.25;v_max=1;t_sway=5;distance=6;
[time2]=func_n(a_max,v_max,t_sway,distance)
time1+time2