function [y, t, y_individual_model] = run_fuzzy_model(u,models, act_table_u, act_table_mu)
%RUN_FUZZY_MODEL Run a Takagi-Sugeno fuzzy model, composed of `tf` object
%linear models. Uses a lookup table to determine the activation of a
%particular model for a particular input value. Each linear model is
%expected to be accurate only in the region where it has high activation
%
%   Input parameters:
%       a               Input signal, array of size 1 by N
%       models          An array of tf object discrete linear models, of
%                       length n.
%       act_table_u     The input part of the activation lookup table, an
%                       array of size 1 by m.
%       act_table_mu    The output part of the activation table, a matrix
%                       of size n by m. Contains mu lookup values for each
%                       model.
%
%   Outputs:
%       y                   Output of the model
%       t                   Time array for the model output
%       y_individual_model  Weighted outputs of individual linear models. 

    num_models = length(models);

    y_individual_model = [];

    for i = 1:num_models
        % Generate activation weights
        w = interp1(act_table_u, act_table_mu(i,:), u, 'cubic');
    
        % Calculate model output
        [y_raw, t] = lsim(models(i), u);
        
        % Apply activation weights
        y_weighted = y_raw.*w';

        y_individual_model = [y_individual_model, y_weighted];
    end
    
    % Merge model outputs
    y = sum(y_individual_model, 2);

end

