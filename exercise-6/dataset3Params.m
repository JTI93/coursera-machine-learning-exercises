function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


% Need to loop through various C and sigma values
% (these are recommended in the exercise text)
values = [ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% Variable within which to store C, sigma and error values
results = [];

for idx1 = 1:length(values),
  for idx2 = 1:length(values),
    % Assign values to our parameters
    test_C = values(idx1);
    test_sigma = values(idx2);
    % Train the model
    model = svmTrain(X, y, test_C, @(x1, x2) gaussianKernel(x1, x2, test_sigma));
    % Predict outputs for the cross-validation set
    predictions = svmPredict(model, Xval);
    % Calculate an error for this model
    error = mean(double(predictions ~= yval));
    % Append the parameters and associated output to the results variable
    results = [results; [test_C, test_sigma, error]];
  endfor
endfor

% Find the index with the lowest error.
[min_err, min_err_idx] = min(results(:,3));

% Choose the parameters associated with the lowest error to be output.
C = results(min_err_idx, 1);
sigma = results(min_err_idx, 2);

% =========================================================================

end