function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%`
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% Need to ass bias layer to a1 ( == X) and a2 after calc
a1 = [ones(m)(:,1), X];
a2 = sigmoid(a1 * Theta1');
a2 = [ones(m)(:,1),a2];
a3 = sigmoid(a2 * Theta2');

% need y to be a 5000 x 10 matrix, so below code converts labels to vectors
% create a column vector of all possible labels
labelVec = zeros(num_labels,1);
for i = 1:num_labels,
    labelVec(i, 1) = i;
end;

% transpose the labelVec vector and repeat both vectors then compare them
% need to convert y into its vector form so it becomes a 5000x10 matrix
checkVec = repmat(labelVec', m, 1);
yWide = repmat(y, 1, num_labels);
yVecs = checkVec == yWide;

% regularize calcs (do not regularize bias terms in Thetas
Theta1sq = Theta1(:,2:end) .* Theta1(:,2:end);
Theta2sq = Theta2(:,2:end) .* Theta2(:,2:end);
reg_term = (sum(sum(Theta1sq)) + sum(sum(Theta2sq))) * lambda/(2*m);
J = (-1/m) * sum(sum((yVecs .* log(a3) + (1 - yVecs) .* log(1 - a3)))) + reg_term;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

d3 = a3 - yVecs;
d2 = d3 * Theta2 .* a2 .* (1 - a2);
% remove first column as this relates to the bias term
% If I calculated d2 using the real sigmoidGradient ( g'(z)= g(z) + (1 - g(z)) )
% then the first column would not have to be removed
d2 = d2(:,2:end);

% This is capital delta in the notes (?)
% Calculate the regularization term, set the first column to 0 (no reg for bias terms)
% Then calculate the gradient for Theta1 ( = D1)

% BACKPROPAGATION NOT CORRECTLY CALCULATING CURRENTLY
Delta1_reg = lambda * Theta1;
Delta1_reg(:,1) = 0;
Delta1 = d2' * a1 + Delta1_reg;
D1 = Delta1/m;

Delta2_reg = lambda * Theta2;
Delta2_reg(:,1) = 0;
Delta2 = d3' * a2 + Delta2_reg;
D2 = Delta2/m;

% assign to output variables
Theta1_grad = D1;
Theta2_grad = D2;
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
