clc; close all; clear all;

%zaèetni pogoj:
 x = [0 0];
 ts = 0.01;%tega se ne da spreminjati!
 ym(1) = 0;
 
 t = 0:ts:15;
 u = 1+0.5*sin(t);

for i = 1:length(t)

    %vhod:
    Fm = u(i);   
    [fi_ fip_] = helicrane(Fm,x);
    x = [fip_ fi_];
    kot(i+1) = fi_; %fi_ je izhod procesa, ki nas zanima.
end
figure
plot(kot)