function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% First, calculate J.
% Since the first term in X does not get regularized,
% we need to not include the first row of theta.
% Create a separate theta for the second term:
theta_J = theta(2:end, :);
% Calculate the cost
J = 1/(2*m) * (X * theta - y)' * (X * theta - y) + lambda/(2*m) * theta_J' * theta_J;

% Now, calculate the gradient.
% Remember that the theta0 term does not get regularized,
% so we create a new theta with zeros in the first row.

% Get the number of columns in theta
cols = size(theta)(2);
% Add a row of zeroes to make the new theta for the regularization term
theta_grad = [zeros(cols); theta_J];
% Calculate the gradient
grad = 1/m * X' * (X * theta - y) + lambda/m * theta_grad;


% NOTE: This has been run through the ex5 and the test cases provided for% this
% script. This code should be fully functional.


% ===========S==============================================================

grad = grad(:);

end
