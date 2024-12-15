function [kot] = proces(u, t, DT)
%PROCES Summary of this function goes here
%   Detailed explanation goes here
    %začetni pogoj:
     x = [0 DT]; % Začetno stanje

    for i = 1:length(t)
    
        %vhod:
        Fm = u(i);   
        [fi_ fip_] = helicrane(Fm,x);
        x = [fip_ fi_];
        kot(i+1) = fi_; %fi_ je izhod procesa, ki nas zanima.
    end

    
end

