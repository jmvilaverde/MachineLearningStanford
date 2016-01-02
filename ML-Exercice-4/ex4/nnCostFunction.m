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
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

%add aditional theta column
completeX = [ones(size(X,1),1) X];

%calculate sigmoid for hidden layer
a2 = sigmoid(completeX*Theta1');

%add aditional theta column
a2 = [ones(size(a2,1),1) a2];

% Get sigmoid per final layer
h = sigmoid(a2*Theta2');

% Transform y, creating an matrix of zeros and setting
% the 1 value depending on y value position

transY = zeros(size(y,1), size(Theta2,1));

for i = 1:size(y,1)

  transY(i, y(i,:)) = 1;

endfor


% Calculate Cost
%J = (1/m) * sum(sum( ((-transY).*log(h)) - ((1-transY).*log(1-h)) ));

%Implement Regularized cost

J = (1/m) * sum(sum( ((-transY).*log(h)) - ((1-transY).*log(1-h)) )) + ( (lambda/(2*m)) * ( sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) ))

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

%D1 = zeros(size(a2,2)-1,size(X,2));
%D2 = zeros(size(h,2),size(a2,2)-1);

D1 = zeros(size(a2,2)-1,size(X,2)+1);
D2 = zeros(size(h,2),size(a2,2));

%one iteration per example
for t = 1:m
  % value per each layer
  a_1 = completeX(t,:);
  a_2 = a2(t,:);
  a_3 = h(t,:);
  ym  = transY(t,:);

  %get delta 3, check if a3 and y contains the same values per each case
  delta3 = (a_3 - ym)';
  
  delta2 = Theta2' * delta3;%
  %remove 0 elements bias
  delta2 = delta2(2:end);
  delta2 = delta2' .* sigmoidGradient(a_1*Theta1');
    
  %D1 = D1 + delta2'*a_1(:,2:end);
  D1 = D1 + delta2'*a_1;
  %size(delta2')
  %size(a_1)
  %size(D1)
  
  %D2 = D2 + delta3*a_2(:,2:end);
  D2 = D2 + delta3*a_2;

endfor

finalD1 = (1/m) * D1;
%size(finalD1)
finalD2 = (1/m) * D2;
%size(finalD2)

%Theta1_grad = [zeros(size(finalD1,1),1) finalD1];
%Theta2_grad = [zeros(size(finalD2,1),1) finalD2];

Theta1_grad = finalD1;
Theta2_grad = finalD2;


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
