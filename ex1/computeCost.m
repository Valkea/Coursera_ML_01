function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

for i = 1:m
	J += ((theta(1)*X(i,1) + theta(2)*X(i,2)) - y(i)).^2;
	% fprintf("J %d: %f -- %f * %f + %f * %f - %f\n", i, J, theta(1), X(i,1), theta(2), X(i,2), y(i));
end

J *= 1/(2*m);
% fprintf("Jfinal: %f\n", J);

% =========================================================================

end
