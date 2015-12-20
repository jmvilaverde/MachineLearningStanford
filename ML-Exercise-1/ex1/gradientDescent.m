function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    H = (X*theta); %Calculate h(theta
    D = H-y; %Difference between expected and actual
    derJ1 = (1/m)*sum(D); %calculate sum for X0
    derJ2 = (1/m)*sum((D.*X(:,2))); %calculate sum for X1
    
    theta(1) = theta(1) - alpha*derJ1
    theta(2) = theta(2) - alpha*derJ2

    %Optimal solution
    %derJ = (1/m)*sum(D.*X); %calculate sum for X0
    %theta = theta - alpha*derJ'
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    fprintf('Cost function values %f\n',J_history(iter))
end

end
