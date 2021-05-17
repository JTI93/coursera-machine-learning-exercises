function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%
%       for i = 1:length(lambda_vec)
%           lambda = lambda_vec(i);
%           % Compute train / val errors when training linear 
%           % regression with regularization parameter lambda
%           % You should store the result in error_train(i)
%           % and error_val(i)
%           ....
%           
%       end
%
%

% The aim of this code is for all lambda values,
% train a model with that value of lambda, calculate
% the error associated with each lambda by using the
% cross-validation set and choose the value of lambda
% that provides the lowest cross-validation error.
% Then, the error of the model should be calculated for
% the test data set.

for i = 1:length(lambda_vec),
    % Set lambda for this loop
    lambda_loop = lambda_vec(i);
    % Train the model for the value of lambda to get theta
    theta_loop = trainLinearReg(X, y, lambda_loop);
    
    % For calculating the error for a trained model,
    % WE SET LAMBDA = 0. Lambda is only used for
    % training a model!
    % Calculate error on training set
    [J_train, grad_train] = linearRegCostFunction(...
        X, y, theta_loop, 0);
    % Calculate error on validation set
    [J_val, grad_val] = linearRegCostFunction(...
        Xval, yval, theta_loop, 0);
    % Assign error values to their vector positions
    error_train(i) = J_train;
    error_val(i) = J_val;
end






% =========================================================================

end
