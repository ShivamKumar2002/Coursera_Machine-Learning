function [J, grad] = nnCostFunction(nn_params, ...
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
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


activation_l1 = [ones(m, 1) X];

y_matrix = zeros(num_labels, m);

for i = 1:m,
    y_matrix(y(i),i) = 1;
end;

activation_l2 = sigmoid(Theta1 * activation_l1');

activation_l2 = [ones(1, size(activation_l2,2));activation_l2];

hypothesis = sigmoid(Theta2 * activation_l2);

term1 = (-y_matrix) .* log(hypothesis);

term2 = (1 - y_matrix) .* log(1 - hypothesis);

J = sum(sum((term1 - term2)))./m;

reg_term = (lambda * (sum(sum(Theta1(:,[2:size(Theta1,2)]).^2)) + sum(sum(Theta2(:,[2:size(Theta2,2)]).^2)))) / (2 * m);

J = J + reg_term;

% -------------------------------------------------------------

deltafunc_3 = hypothesis - y_matrix;

sg = sigmoidGradient(Theta1 * activation_l1');

sg = [ones(1, size(sg,2)) ; sg];

deltafunc_2 = (Theta2' * deltafunc_3) .* sg;

deltafunc_2 = deltafunc_2(2:size(deltafunc_2,1), :);

accumulator1 = deltafunc_2 * activation_l1;

accumulator2 = deltafunc_3 * activation_l2';

Theta1_grad = accumulator1 ./ m;

Theta2_grad = accumulator2 ./ m;


% =========================================================================


t1 = Theta1(:,2:size(Theta1,2));
t2 = Theta2(:,2:size(Theta2,2));

t1 = [zeros(size(t1,1),1),t1];
t2 = [zeros(size(t2,1),1),t2];

reg_term_grad_1 = (lambda / m) .* t1;
reg_term_grad_2 = (lambda / m) .* t2;


Theta1_grad = Theta1_grad + reg_term_grad_1;
Theta2_grad = Theta2_grad + reg_term_grad_2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
