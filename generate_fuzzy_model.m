function [models] = generate_fuzzy_model(u,y, ts, act_table_u, act_table_mu)
%GENERATE_FUZZY_MODEL] Identify linear model params to use in the TS model.
%   Generate linear ARX models (without the stochastic part) for each cluster.
%   We perform local optimization using Weighted Linear Least Squares.
%   Local optimization is simpler and provides better interpretation than
%   global optimization, although resulting in slightly worse performance.
%   The activation table is used to generate the weights matrix for the
%   WLLS.
%
%   The number of models `n` is determined by the size of the activation
%   table.
%
%   We hardcode this to a second order discrete linear model, with one
%   zero:
%
%       b1 z^-1 + b2 z-2              b1 z^1 + b2
%   ------------------------  =  --------------------
%       1 + a1 z^-1 + a2 y^-2        z^2 + a1 z^1 + a2
%
%   Input parameters:
%       u           Input training signal
%       y           Output training signal
%       ts          Sample time
%       act_table_u     The input part of the activation lookup table, an
%                       array of size 1 by m.
%       act_table_mu    The output part of the activation table, a matrix
%                       of size n by m. Contains mu lookup values for each
%                       model.
%
%   Output parameters:
%       models      A 1 by n array of discrete linear models (ts objects).

    num_clusters = size(act_table_mu,1);

    model_weights = [];
    models = [];
    
    for i = 1:num_clusters
        % Generate diagonal weights matrix based on the i-th cluster activation function
        % Performs table lookup (outputs weigths for each sample in the input
        % signal)
        W = interp1(act_table_u, act_table_mu(i,:), u(3:end));
        W = sparse(1:length(W), 1:length(W), W); % Use a sparse diagonal matrix to save memory
    
        % Generate the data matrix with delayed inputs for a second order
        % one zero model.
        m = 2;
        N = length(y);
        X = [u(2:N-1)', u(1:N-m)', -y(2:N-1)', -y(1:N-2)'];
    
        % Perform WLS
        theta_hat = (X'*W*X)\X'*W*y(m+1:end)';
    
        % discrete-time TF
        model_weights(:,i) = theta_hat;
    
        num = theta_hat(1:2)';
        denom = [1, theta_hat(3:4)'];
        models = [models, tf(num,denom,ts)];
    end
end

