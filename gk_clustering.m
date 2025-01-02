function [centers,cov,x_grid,y_grid,val_grid] = gk_clustering(X,c,eta,grid_accuracy,stop_threshold,max_iter)
%GK_CLUSTERING Gustafson-Kessel fuzzy clustering
%   Kristjan Soln, 2024
%   
%   Performs fuzzy clustering using the Gustafson-Kessel method. Using
%   Mahalanobis norm for measuring distance between points, it can adapt
%   cluster shapes to follow the data. This results in elipsoid clusters, 
%   rotated and with both axes scaled accordingly.
%
%   This version is modified to be less sensitive to different cluster
%   volumes (different number of points in each cluster). It calculates
%   cluster volume L_i and uses it in the limitation factor rho_i.
%
%   Can be unstable, try rerunning in that case. Normalizing the data might
%   also help.
%
%   This method is sensitive to outliers, as they can deform the clusters
%   significantly. 
%
%   Works with arbitrary number of dimensions. If data is two-dimensional,
%   also calculates values related to grid of assignment values, used to
%   visualize a contour plot of clusters. Use 
%   contourf(x_grid,y_grid,val_grid) for that.
%
%   Input parameters:
%       X - input data matrix, with data in columns. Preferably 
%             normalized.
%       c - number of clusters, with default value of 3
%       eta - fuzziness coefficient, can be from 1 to Inf. Determines the
%             width/fuzziness of the clusters. Value of 1 means crisp 
%             clusters and Inf means infinitely fuzzy clusters (all samples
%             belonging to all clusters with the same degree).
%             Default value is 2.
%       grid_accuracy - the precision of the grid for calculating
%             assignment values and ploting cluster contours. Defaults to
%             30.
%       stop_threshold - minimum change in cluster center positions.
%             Defaults to 0.001.
%       max_iter - max number of iterations. Defaults to 100.
% 
%   Outputs:
%       centers - cluster centers
%       cov - cluster covariance matrices (maybe)
%
%   Additional outputs related to grid of assignment values:
%       x_grid - x coordinates of the grid
%       y_grid - y coordinates of the grid
%       val_grid - grid values matrix

    arguments
        X
        c = 3
        eta = 2
        grid_accuracy = 30
        stop_threshold = 0.001
        max_iter = 100
    end

    N = length(X); % Number of samples
    n = length(X(1,:)); % Number of variables

    % Define cluster starting positions with random samples
    % v = X(round(rand([c,1])*(length(X)-1) + 0.5), :);
    % OR Define cluster starting positions with equal intervals across the
    % axes.
    % This is good for clustering the input-output space.
    top_boundary = max(X(:,1));
    bot_boundary = min(X(:,1));
    delta_v = (top_boundary-bot_boundary)/(c-1);
    v_u = [bot_boundary:delta_v:top_boundary];

    top_boundary = max(X(:,2));
    bot_boundary = min(X(:,2));
    delta_v = (top_boundary-bot_boundary)/(c-1);
    v_y = [bot_boundary:delta_v:top_boundary];

    v = [v_u', v_y'];

    % Init the U matrix, normalize it so columns sum up to 1
    % This gives the samples a random assignment to clusters
    U = rand([c,N]);
    U = U./repmat(sum(U),c,1);

    % Initialize the inner product matrix F
    F = [];


    % Distance function
    % Generalized square norm
    % Inner product matrix is calculated during the algorithm based on the
    % current state
    d = @(x, v, A) (x-v)*A*(x-v)';

    v_hist = v; % For plotting
    iteration = 0;

    disp("Starting the clustering algorithm")

    while(true)
        iteration = iteration + 1;
        if mod(iteration,5) == 0
            disp("Iteration " + iteration)
        end

        % Update center for each cluster
        % This is a normalized weighted sum of all samples, weight is based on
        % mu of that sample
        for i = 1:c
            center = (U(i,:).^eta * X) / sum(U(i,:).^eta);
            v(i,:) = center;
        end

        % Store the centers for plotting
        v_hist(:,:,iteration+1) = v;

        cluster_covariances = [];


        for i = 1:c
            % Calculate the fuzzy covariance matrix for each cluster
            sigma_i = zeros(n);
            for k = 1:N
                mu = U(i,k);
                vi = v(i,:);
                xk = X(k,:);
                sigma_i = sigma_i + mu^eta*(xk-vi)'*(xk-vi);
            end
            sigma_i = sigma_i / sum(U(i,:));
            % Calculate the inner product matrix F for each cluster
            L_i = sum(U(i,:));  % Cluster volume
            rho_i = 1/L_i; % Limitation factor
            F(:,:,i) = (rho_i * det(sigma_i))^(1/n) * inv(sigma_i);

            % For outputting cluster covariances at the end
            cluster_covariances(:,:,i) = sigma_i;
        end

        % Calculate distances of each point to each cluster
        D = zeros(N,c);
        for i = 1:c
            center = v(i,:);
            for k = 1:N
                D(k,i) = d(X(k,:), center, F(:,:,i));
            end
        end


        % Update the assignment matrix U
        w = 2/(eta-1);
        for i = 1:c
            for k = 1:N
                Di_sum = sum(1./(D(k,:).^w));
                Di = D(k,i);
                U(i,k) = (1/Di^w) / Di_sum;
            end
        end


        % Check stop conditions
        if(all(abs(v_hist(:,:,end) - v_hist(:,:,end-1)) < stop_threshold, 'all'))
            disp("Stopping in iteration " + string(iteration) + " due to stop threshold reached")
            break
        end
        if(iteration >= max_iter)
            disp("Stopping due to max iterations reached")
            break
        end
    end

    % Calculate the U grid for contours, only if data is two dimensional
    if n == 2
        % You basically need to create a grid of points and calculate the
        % assignment values to clusters. Then, pick the maximum one for each grid
        % point and display it.
        step = (max(X(:,1)) - min(X(:,1)))/grid_accuracy;
        x_grid = min(X(:,1)):step:max(X(:,1));
        step = (max(X(:,2)) - min(X(:,2)))/grid_accuracy;
        y_grid = min(X(:,2)):step:max(X(:,2));
        % Calculate distances of each grid point to each cluster center
        D = zeros(length(x_grid),length(y_grid), c);
        for i = 1:c
            center = v(i,:);
            for xi = 1:length(x_grid)
                for yi = 1:length(y_grid)
                    x = x_grid(xi);
                    y = y_grid(yi);
                    D(xi,yi,i) = d([x y], center, F(:,:,i));
                end
            end
        end
        % Update the assignment matrix U
        U = zeros(length(x_grid), length(y_grid), c);
        w = 2/(eta-1);
        for i = 1:c
            for xi = 1:length(x_grid)
                for yi = 1:length(y_grid)
            % for k = 1:N
                Di_sum = sum(1./(D(xi,yi,:).^w));
                Di = D(xi,yi,i);
                U(xi,yi,i) = (1/Di^w) / Di_sum;
                end
            end
        end
        % Get the largest assigment value for each point
        U_tight = zeros(length(x_grid), length(y_grid));
        for xi = 1:length(x_grid)
            for yi = 1:length(y_grid)
                U_tight(xi,yi) = max(U(xi,yi,:));
            end
        end
    
    else
        x_grid = -1;
        y_grid = -1;
        U_tight = -1;
    end

    % Output vars
    centers = v;
    cov = cluster_covariances;
    x_grid = x_grid;
    y_grid = y_grid;
    val_grid = U_tight';

end

