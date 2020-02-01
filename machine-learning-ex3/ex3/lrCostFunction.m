function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% step: find h_theta(x)
%
h = sigmoid(X * theta);

% step: find the average cost of h before regularization 
%
cost_avg = 1/m * sum((y .* (-log(h))) + ((1-y) .* (-log(1-h))));

% note:
% some people "simplify" this by moving the signs of the main cost function to: 
% cost_per = -y.*log(h)-(1-y).*log(1-h);
%            ^         ^

% step: find the regularization value
%
% theta is indexed from 0, here we explicitly remove the first value
% because we do not regularize the bias theta value??
regularization = (lambda/(2*m)) * sum(theta(2:end).^2);

% step: find J which includes cost and regularization
#
J = cost_avg + regularization

% step: find the gradient
%
% method 1: slice the theta vector and then add back in the bias factor
%
grad_wo_reg =  1/m * (X' * (h - y));
grad_reg_wo_bias = (lambda/m) * theta(2:end);
grad_reg = [0; grad_reg_wo_bias];
grad = grad_wo_reg + grad_reg;

% method 2: copy the theta vector and set the first item to 0
%
grad_unregularized = 1/m * (X' * (h - y));
temp = theta;
temp(1) = 0;
grad_regularization = (lambda/m) * temp;
grad = grad_unregularized + grad_regularization

% =============================================================

grad = grad(:);

end
