%% ASSIGNMENT 5
% Nonlinear dynamical model
% Kristjan Šoln

% ts = 0.01; %tega se ne da spreminjati!
% 
% t = 0:ts:15;
% u = sin(t);
% 
% kot = proces(u, t, 0);
% plot(t, kot(1:end-1))

%% Generate training and validation input signal for identification and perform measurements

clc; clear all; close all;

% ts = 0.01;  % Sampling time
ts = 0.01;
umin = 0;
umax = 1.35;

% Training signal

u_train = [];
y_train = [];
t_train = [];

% Step signal
n_steps = 20;
step_time = 7;
deltau = (umax-umin)/n_steps;

for i = 0:n_steps
    t_new = (i*step_time):ts:((i+1)*step_time);
    u_new = (umin + i*deltau);
    
    t_train = [t_train, t_new];
    u_train = [u_train, u_new * ones(size(t_new))];
end
for i = 1:n_steps
    t_new = t_train(end):ts:(t_train(end)+step_time);
    u_new = (umax - i*deltau);
    
    t_train = [t_train, t_new];
    u_train = [u_train, u_new * ones(size(t_new))];
end

% PBRS signal
pbrs_time = 500;
t_new = t_train(end):ts:(t_train(end)+pbrs_time);
u_new = umin + (umax-umin)*round(rand(size(t_new)));
% u_new = umin + (umax-umin)*(rand(size(t_new))); % aprbs
t_train = [t_train, t_new];
u_train = [u_train, u_new];

% Output signal of the process
y_train = proces(u_train,t_train,0);
y_train = y_train(1:end-1);

% Validation signal

u_valid = [];
y_valid = [];
t_valid = [];

% Step signal
n_steps = 10;
step_time = 10;
deltau = (umax-umin)/n_steps;

for i = 0:n_steps
    t_new = (i*step_time):ts:((i+1)*step_time);
    u_new = (umin + i*deltau);
    
    t_valid = [t_valid, t_new];
    u_valid = [u_valid, u_new * ones(size(t_new))];
end
for i = 1:n_steps
    t_new = t_valid(end):ts:(t_valid(end)+step_time);
    u_new = (umax - i*deltau);
    
    t_valid = [t_valid, t_new];
    u_valid = [u_valid, u_new * ones(size(t_new))];
end

% PBRS signal
pbrs_time = 100;
t_new = t_valid(end):ts:(t_valid(end)+pbrs_time);
% u_new = umin + (umax-umin)*round(rand(size(t_new)));
u_new = umin + (umax-umin)*(rand(size(t_new))); % aprbs
t_valid = [t_valid, t_new];
u_valid = [u_valid, u_new];

% Output signal
y_valid = proces(u_valid,t_valid,0);
y_valid = y_valid(1:end-1);

% Plot the train and validation signals
figure();
subplot(2,1,1);
plot(t_train, u_train);
title("Training input signal")
xlabel("t"); ylabel("u(t)");

subplot(2,1,2);
plot(t_train, y_train)
title("Training output signal");
xlabel("t"); ylabel("y(t)")

figure();
subplot(2,1,1);
plot(t_valid, u_valid);
title("Validation input signal")
xlabel("t"); ylabel("u(t)");

subplot(2,1,2);
plot(t_valid, y_valid)
title("Validation output signal");
xlabel("t"); ylabel("y(t)")

