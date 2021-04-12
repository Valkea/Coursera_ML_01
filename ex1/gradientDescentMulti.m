function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %
  num_cols_X = size(X,2);
  gradient = zeros(num_cols_X,1);

  for i = 1:m

		thetaSum = 0;
		for j = 1:num_cols_X
			thetaSum += theta(j)*X(i,j);
		end

		for k = 1:num_cols_X
                	gradient(k) += (thetaSum - y(i))*X(i,k);
		end
        
  end

  theta -= alpha * (1/m) * gradient;

% Declare convergeance if J(Θ) decreases by less than 10⁻³ in one iteration
% For sufficiently small α, J(Θ) should decrease on every iteration.
% But if α is too small, gradient descent can be slow to converge.
% And if α is too large, J(θ) may not decrease on every iteration and converge.

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
