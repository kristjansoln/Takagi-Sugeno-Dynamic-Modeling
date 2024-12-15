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

%% Neural network model

% We use a time-series neural network model, a nonlinear ARX model.
% Tutorial:
% https://www.mathworks.com/help/deeplearning/ug/design-time-series-narx-feedback-neural-networks.html

U = num2cell(u_train);
Y = num2cell(y_train);

% Perform the training in an open-loop configuration
input_delays = 1:2;
feedback_delays = 1:2;
model_size = 10;
net = narxnet(input_delays, feedback_delays, model_size);
net.divideFcn = '';
net.trainParam.min_grad = 1e-10;
[p,Pi,Ai,t] = preparets(net,U,{},Y);

net = train(net,p,t,Pi);

% Now that the training is over, close the loop
net_closed = closeloop(net);
% Perform validation
U_valid = num2cell(u_valid);
Y_valid = num2cell(y_valid);
[inputs,Pi1,Ai1,t1] = preparets(net_closed,U_valid,{},Y_valid);
% [inputs,Pi1,Ai1] = preparets(net_closed,U_valid,{});

% Gets model output
y_hat_nn = net_closed(inputs,Pi1,Ai1);
y_hat_nn = cell2mat(y_hat_nn);

% Calculate statistics, plot model output
e = y_hat_nn - y_valid(3:end);
rms_error = rmse(y_hat_nn, y_valid(3:end));
disp("Root Mean Square error: " + string(rms_error));
disp("Standard deviation of error: " + string(std(e)))

figure();
subplot(2,1,1);
plot(t_valid(3:end), y_valid(3:end));
hold on;
plot(t_valid(3:end), y_hat_nn);
title("Model output")
legend("True value", "Model output")
xlabel("t"); ylabel("y(t)");

subplot(2,1,2);
plot(t_valid(3:end), e)
title("Error through time");
xlabel("t"); ylabel("e(t)")
