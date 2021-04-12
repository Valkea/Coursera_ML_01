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
%	if( computeCost(X, y, theta) < 0 )
%		break;
%	end

	gradient = zeros(2,1);

	for i = 1:m
		%gradient += ((theta(1)*X(i,1) + theta(2)*X(i,2)) - y(i))*X(i);
		gradient(1) += ((theta(1)*X(i,1) + theta(2)*X(i,2)) - y(i));
		gradient(2) += ((theta(1)*X(i,1) + theta(2)*X(i,2)) - y(i))*X(i,2);
	end

	theta -= alpha * (1/m) * gradient;

% Declare convergeance if J(Θ) decreases by less than 10⁻³ in one iteration
% For sufficiently small α, J(Θ) should decrease on every iteration.
% But if α is too small, gradient descent can be slow to converge.
% And if α is too large, J(θ) may not decrease on every iteration and converge.


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
