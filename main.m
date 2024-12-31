%% ASSIGNMENT 5

% Nonlinear dynamical model
% Kristjan Šoln

%% Generate training and test input signal for identification and perform measurements

clc; clear all; close all;
disp("Generate training and test signals")

ts = 0.01;  % Sampling time
umin = 0;
umax = 1.35; % Saturates the output

% Training signal

u_train = [];
y_train = [];
t_train = [];

% Step signal
n_steps = 20;
step_time = 6;
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
pbrs_time = 500;
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
n_steps = 20;
step_time = 10;
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

% Gets model output
y_hat_nn = net_closed(inputs,Pi1,Ai1);
y_hat_nn = cell2mat(y_hat_nn);

% Calculate statistics, plot model output
e = y_hat_nn - y_test(3:end);
rms_error = rmse(y_hat_nn, y_test(3:end));
disp("Root Mean Square error: " + string(rms_error));
disp("Standard deviation of error: " + string(std(e)));

figure();
subplot(2,1,1);
plot(t_test(3:end), y_test(3:end));
hold on;
plot(t_test(3:end), y_hat_nn);
title("Neural network model: output")
legend("True value", "Model output")
xlabel("t"); ylabel("y(t)");

subplot(2,1,2);
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

disp("Fuzzy model")

num_clusters = 5;
cluster_fuzziness = 1.9;
clustering_iterations = 30;

% Prepare the input output space for clustering
% We perform the clustering without the APRBS part, as it seems to mess
% with the clustering algorithm.
X = [u_train_noprbs', y_train_noprbs'];

% Perform clustering
[centers, cov_centers, x_grid, y_grid, val_grid] = gk_clustering(X,num_clusters,cluster_fuzziness,400,0.001,clustering_iterations);

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
    std_u = sqrt(cov_centers(1,1,i));
    act_table = [act_table, gaussmf(act_table_u, [std_u,mean_u])']; % The dependent variable in the lookup table
end

% Normalize the activation functions
act_table_norm = act_table./repmat(sum(act_table, 2), 1,num_clusters);

% Plot the activation functions
subplot(2,2,3)
hold on; xlabel("input - u"); ylabel("activation function value - \mu_i");
title("Takagi-Sugeno activation function values")
for i = 1:num_clusters
    plot(act_table_u, act_table(:,i));
end
subplot(2,2,4);
hold on; xlabel("input - u"); ylabel("cluster membership - \mu_i");
title("Takagi-Sugeno activation function values - normalized")
for i = 1:num_clusters
    plot(act_table_u, act_table_norm(:,i));
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


% Determine U0 and Y0 operating points
cluster_operating_points = [];
for i = 1:num_clusters
    U0 = centers(i,1);

    ts = 0.01;
    t = 0:ts:30;
    y = proces(ones(size(t))*U0,t,0);
    Y0 = y(end);
    cluster_operating_points = [cluster_operating_points; U0, Y0];
end

clear y t

model_weights = [];
models = [];

for i = 1:num_clusters
    % Generate diagonal weights matrix based on the i-th cluster activation function
    % Performs table lookup (outputs weigths for each sample in the input
    % signal)
    W = interp1(act_table_u, act_table_norm(:,i), u_train(3:end));
    W = sparse(1:length(W), 1:length(W), W); % Use a sparse matrix to save memory

    % Correct the input and output signals to represent deviations from the
    % operating point
    y = y_train - cluster_operating_points(i,2);
    u = u_train - cluster_operating_points(i,1);

    % Generate the psi matrix with delayed inputs
    m = 2;
    N = length(y);
    X = [u(m:N-1)', u(m-1:N-m)', -y(m:N-1)', -y(1:N-m)'];

    % Perform WLS
    theta_hat = (X'*W*X)\X'*W*y(m+1:end)';

    % discrete-time TF
    model_weights(:,i) = theta_hat;

    num = theta_hat(1:2)';
    denom = [1, theta_hat(3:4)'];
    models = [models, tf(num,denom,ts)];
end

clear W y u X num denom

% Model evaluation
% Perform model evaluation using the test signal

individual_model_output = [];

for i = 1:num_clusters
    % Generate weights matrix
    w = interp1(act_table_u, act_table_norm(:,i), u_test(3:end));

    % Correct the input signal to represent deviations from the operating point
    u_rel = u_test - cluster_operating_points(i,1);

    % Calculate model output
    [y,t_out] = lsim(models(i), u_rel(3:end));
    y = y + cluster_operating_points(i,2);  % Take operating point into account
    y_weighted = y.*w';
    individual_model_output = [individual_model_output, y_weighted];
end

% Merge model outputs
y_hat_fuzzy = sum(individual_model_output, 2);

% Plot the output
figure();
title("Fuzzy model evaluation: Individual Model Output");
hold on; grid on;
for i = 1:num_clusters
    plot(t_out, individual_model_output(:,i))
end
plot(t_test, y_test, '-')
legend("Model 1","Model 2","Model 3","Model 4","Model 5","y_{test}")
xlabel("t");

figure();
title("Fuzzy model evaluation: Summed output")
hold on; grid on;
plot(t_test, y_test, '-')
plot(t_test(3:end), y_hat_fuzzy)
legend("$y_{test}$", '$\hat{y}_{test}$' ,'Interpreter','latex')
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

disp(" ")
