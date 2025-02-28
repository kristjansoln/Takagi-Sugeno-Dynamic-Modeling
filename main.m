%% ASSIGNMENT 5

% Nonlinear dynamical model
% Kristjan Šoln

%% Generate training and test input signal for identification and perform measurements

clc; clear all; close all;
disp("Generate training and test signals")

ts = 0.01;  % Sampling time
umin = 0;
umax = 1.35; % Saturates the output
% umax = 1.32;

% Training signal

u_train = [];
y_train = [];
t_train = [];

% Step signal
n_steps = 30;
step_time = 10;
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

% Store the signal without the PRBS part, for GK clustering
t_train_noprbs = t_train;
u_train_noprbs = u_train;

% APBRS signal
% APRBS should contain pulses with lenght of at least the rise time of the
% process. Here, 1.5 or 2 sec should be enough.
pbrs_time = 1500;
t_new = t_train(end):ts:(t_train(end)+pbrs_time);

len_min = round(2/ts); % In samples
len_max = len_min*3;
k = 1;
while k < length(t_new)
    len = len_min + round(rand()*len_max);
    val = umin + (umax-umin)*(rand());
    
    u_new = [u_new, repmat(val, [1,len])];
    k = k+len;
end

% Trim the aprbs signal to the correct length
u_new = u_new(1:length(t_new));

t_train = [t_train, t_new];
u_train = [u_train, u_new];

% Output signal of the process
y_train = proces(u_train,t_train,0);
y_train = y_train(1:end-1);
y_train_noprbs = proces(u_train_noprbs, t_train_noprbs, 0);
y_train_noprbs = y_train_noprbs(1:end-1);

% Test signal

u_test = [];
y_test = [];
t_test = [];

% Step part
n_steps = 12;
step_time = 13;
deltau = (umax-umin)/n_steps;

for i = 0:n_steps
    t_new = (i*step_time):ts:((i+1)*step_time);
    u_new = (umin + i*deltau);
    
    t_test = [t_test, t_new];
    u_test = [u_test, u_new * ones(size(t_new))];
end
for i = 1:n_steps
    t_new = t_test(end):ts:(t_test(end)+step_time);
    u_new = (umax - i*deltau);
    
    t_test = [t_test, t_new];
    u_test = [u_test, u_new * ones(size(t_new))];
end

% APBRS part
pbrs_time = 100;
t_new = t_test(end):ts:(t_test(end)+pbrs_time);

len_min = round(2/ts); % In samples
len_max = len_min*3;
k = 1;
while k < length(t_new)
    len = len_min + round(rand()*len_max);
    val = umin + (umax-umin)*(rand());
    
    u_new = [u_new, repmat(val, [1,len])];
    k = k+len;
end

% Trim the aprbs signal to the correct length
u_new = u_new(1:length(t_new));

t_test = [t_test, t_new];
u_test = [u_test, u_new];

% Output signal

y_test = proces(u_test,t_test,0);
y_test = y_test(1:end-1);

% Plot the train and test signals

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
plot(t_test, u_test);
title("Test input signal")
xlabel("t"); ylabel("u(t)");

subplot(2,1,2);
plot(t_test, y_test)
title("Test output signal");
xlabel("t"); ylabel("y(t)")

disp(" ")

clear deltau i t_new u_new

%% Neural network model

% We use a time-series neural network model, a nonlinear ARX model.
% Tutorial:
% https://www.mathworks.com/help/deeplearning/ug/design-time-series-narx-feedback-neural-networks.html

disp("Neural network model")

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
% Perform testing
u_test_cell = num2cell(u_test);
y_test_cell = num2cell(y_test);
[inputs,Pi1,Ai1,t1] = preparets(net_closed,u_test_cell,{},y_test_cell);
% [inputs,Pi1,Ai1] = preparets(net_closed,U_test,{});

% Get model output
y_hat_nn = net_closed(inputs,Pi1,Ai1);
y_hat_nn = cell2mat(y_hat_nn);

% Calculate statistics, plot model output
e = y_hat_nn - y_test(3:end);
rms_error = rmse(y_hat_nn, y_test(3:end));
disp("Root Mean Square error: " + string(rms_error));
disp("Standard deviation of error: " + string(std(e)));

figure();
plot(t_test(3:end), y_test(3:end));
hold on;
plot(t_test(3:end), y_hat_nn);
title("Neural network model: output")
legend("True value", "Model output")
xlabel("t"); ylabel("y(t)");

figure();
plot(t_test(3:end), e)
title("Neural network model: Error through time");
xlabel("t"); ylabel("e(t)")

disp(" ")

%% Fuzzy model
% We get clusters using Gustafson-Kessel fuzzy clustering to use with the
% Takagi-Sugeno method. Based on those cluster centers and variances, we
% define Gaussian activation functions for the Takagi-Sugeno models. Then,
% we define a linear model for each cluster and calculate the parameters
% using the weigthed least squares method.

disp("Fuzzy model identification")

num_clusters = 5;
cluster_fuzziness = 2.0;%1.9;
clustering_iterations = 100;%30;

% Prepare the input output space for clustering
% We perform the clustering without the APRBS part, as it seems to mess
% with the clustering algorithm.
X = [u_train_noprbs', y_train_noprbs'];

% Perform clustering
[centers, cov_centers, x_grid, y_grid, val_grid] = gk_clustering(X,num_clusters,cluster_fuzziness,200,0.001,clustering_iterations);

% Draw points, cluster centers and membership contours
figure; subplot(2,1,1)
contourf(x_grid,y_grid,val_grid, 'FaceAlpha', 0.25)
hold on;
plot(X(:,1), X(:,2), '.', 'color', 'black')
plot(centers(:,1), centers(:,2), 'pentagram', 'color', 'red')
title("Clustering of the input-output space")
xlabel("input - u"); ylabel("output - y");
legend("membership contour plot", "data in input-output space", "cluster centers", "Location", "northwest")

% Although GK returns a complicated membership function (not a simple
% Gaussian or normalized Gaussian), when defining a TS model, we can assume
% normalized Gaussian activation functions and use only cluster centers
% and the variances from the GK clusters.

% Generate lookup table matrix for getting activation functions from input
% value

act_table = [];

du = (umax-umin)/200;
act_table_u = umin:du:umax; %  The independent variable in the lookup table

for i = 1:num_clusters
    mean_u = centers(i,1);
    dev_u = sqrt(cov_centers(1,1,i));
    act_table = [act_table, gaussmf(act_table_u, [dev_u,mean_u])']; % The dependent variable in the lookup table
end

% Normalize the activation functions
act_table_norm = act_table./repmat(sum(act_table, 2), 1,num_clusters);

act_table = act_table';
act_table_norm = act_table_norm';

% Plot the activation functions
subplot(2,2,3)
hold on; xlabel("input - u"); ylabel("activation function value - \mu_i");
title("Takagi-Sugeno activation function values")
for i = 1:num_clusters
    plot(act_table_u, act_table(i,:));
end
subplot(2,2,4);
hold on; xlabel("input - u"); ylabel("cluster membership - \mu_i");
title("Takagi-Sugeno activation function values - normalized")
for i = 1:num_clusters
    plot(act_table_u, act_table_norm(i,:));
end

clear X x_grid y_grid val_grid act_table

% Generate linear ARX models (without the stochastic part) for each cluster.
% We perform local optimization using weighted linear least squares.
% Local optimization is simpler and provides better interpretation than
% global optimization, although resulting in slightly worse performance.

% We pick a second order model, with one zero.
%     b1 z^-1 + b2 z-2              b1 z^1 + b2
% ------------------------  =  --------------------
%   1 + a1 z^-1 + a2 y^-2        z^2 + a1 z^1 + a2

models = generate_fuzzy_model(u_train, y_train, ts, act_table_u, ...
    act_table_norm);

clear W y u X num denom

% Model evaluation
% Perform model evaluation using the test signal
disp("")
disp("Fuzzy model evaluation:")

[y_hat_fuzzy, t_out, individual_model_output] = run_fuzzy_model( ...
    u_test(3:end), models, act_table_u, act_table_norm);

% Plot the output
figure();
title("Fuzzy model evaluation: Individual Model Output");
hold on; grid on;
for i = 1:num_clusters
    plot(t_out, individual_model_output(:,i))
end
plot(t_test, y_test, '--')
legend("Model 1","Model 2","Model 3","Model 4","Model 5","y_{test}")
xlabel("t");

figure();
title("Fuzzy model evaluation: Summed output")
hold on; grid on;
plot(t_test, y_test, '-')
plot(t_test(3:end), y_hat_fuzzy)
legend("$y_{test}$", '$\hat{y}_{model}$' ,'Interpreter','latex')
xlabel("t");

% Calculate and plot the error
e = y_hat_fuzzy - y_test(3:end)';
rms_error = rmse(y_hat_fuzzy, y_test(3:end)');
disp("Root Mean Square error: " + string(rms_error));
disp("Standard deviation of error: " + string(std(e)));

figure();
plot(t_test(3:end), e)
title("Fuzzy model evaluation: Error through time");
xlabel("t"); ylabel("e(t)"); grid on;

% Save the model
% cov_centers = squeeze(cov_centers(1,1,:));
% save hcr_fuzzy_model_params models act_table_u act_table_norm centers cov_centers

%% Add aditional models for large input values
% Additional models help with prediction near output saturation

%clear all
close all
load hcr_fuzzy_model_params.mat

disp(" ")
disp("Optimize model for large input values")

u_min = 0;
u_max = 1.35;
num_extra_models = 6;
dev_factor = 0.6; % How narrow the new assignment functions will be

% Sort centers and deviations
centers_u = centers(:,1);
[centers_u, i] = sort(centers_u);
dev_u = sqrt(cov_centers(i));

% Remove the highest two centers and add extra new centers up to u_max
disp("Removing centers " + string(centers_u(end)) + " and " + string(centers_u(end-1)));
dev_umax_old = dev_u(end); % Remember the top cluster deviation for later reference

centers_u = centers_u(1:end-2);
dev_u = dev_u(1:end-2);

low_bound = centers_u(end) + 1.5*dev_u(end);
du = (u_max - low_bound)/(num_extra_models - 1);
centers_u = [centers_u; (low_bound:du:u_max)'];
dev_u = [dev_u; dev_factor*dev_umax_old*ones(size((low_bound:du:u_max)'))];

% Generate assignment function table

act_table = [];

du = (u_max-u_min)/400;
act_table_u_opt = u_min:du:u_max; %  The independent variable in the lookup table
num_clusters = length(centers_u);

for i = 1:num_clusters
    mean = centers_u(i);
    dev = dev_u(i);
    act_table = [act_table, gaussmf(act_table_u_opt, [dev,mean])']; % The dependent variable in the lookup table
end

% Normalize the activation functions
act_table_norm = act_table./repmat(sum(act_table, 2), 1,num_clusters);

act_table = act_table';
act_table_norm = act_table_norm';

% Plot the activation functions
figure();
subplot(2,1,1)
hold on; xlabel("input - u"); ylabel("activation function value - \mu_i");
title("Optimized TS activation function values")
for i = 1:num_clusters
    plot(act_table_u_opt, act_table(i,:));
end
subplot(2,1,2);
hold on; xlabel("input - u"); ylabel("cluster membership - \mu_i");
title("Optimized TS activation function values - normalized")
for i = 1:num_clusters
    plot(act_table_u_opt, act_table_norm(i,:));
end

% Regenerate the fuzzy model
models_opt = generate_fuzzy_model(u_train, y_train, ts, act_table_u_opt, ...
    act_table_norm);

% Evaluate the model
disp("Evaluation of the optimized model")

[y_hat_fuzzy_opt, t_out_opt, individual_model_output_opt] = run_fuzzy_model( ...
    u_test(3:end), models_opt, act_table_u_opt, act_table_norm);

figure();
title("Optimized fuzzy model evaluation: output")
hold on; grid on;
plot(t_test, y_test, '-')
plot(t_out_opt, y_hat_fuzzy_opt)
legend("$y_{test}$", '$\hat{y}_{model}$' ,'Interpreter','latex')
xlabel("t");

% Calculate and plot the error
e = y_hat_fuzzy_opt - y_test(3:end)';
rms_error = rmse(y_hat_fuzzy_opt, y_test(3:end)');
disp("Root Mean Square error: " + string(rms_error));
disp("Standard deviation of error: " + string(std(e)));

figure();
plot(t_test(3:end), e)
title("Optimized fuzzy model evaluation: Error through time");
xlabel("t"); ylabel("e(t)"); grid on;

% Save the model
% models=models_opt;
% act_table_u = act_table_u_opt;
% centers = centers_u;
% dev_centers = dev_u;
% save hcr_optimized_model models act_table_u act_table_norm centers dev_centers
