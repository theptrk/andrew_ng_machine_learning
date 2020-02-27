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

% Add bias to X
X = [ones(size(X,1), 1), X];

% Calculate layer2
layer2 = sigmoid(X * Theta1');

% Add bias to layer2
layer2 = [ones(size(layer2, 1), 1), layer2];

% Calculate layer3
layer3 = sigmoid(layer2 * Theta2');

% y is a vector of labels in range [1, 10]
% Convert the labels to 10 dimensional vectors of values in range [0,1] 
% where the index represents the label value. 

% So one is: [1; 0; 0; 0; ... 0]. Two is: [0; 1; 0; 0; ... 0]


% create a store for our label to label vector conversion 
y_with_label_vectors = zeros(10, size(y, 1));

% for every label
for label_index = 1:size(y, 1)
	% create a "label vector"
	% for the index for the label
	label_vector = zeros(10, 1);
	label = y(label_index);
	label_vector(label) = 1;

	% place a "label vector"
	y_with_label_vectors(:, label_index) = label_vector;
end

unique(y_with_label_vectors == convert_10_dim(y))

function [new_matrix] = convert_10_dim(vector_of_num_labels)
    y_with_label_vectors = zeros(10, size(y, 1));

    for label_index = 1:size(y, 1)
        % create a "label vector"
        % for the index for the label
        label_vector = zeros(10, 1);
        label = y(label_index);
        label_vector(label) = 1;

        % place a "label vector"
        y_with_label_vectors(:, label_index) = label_vector;
    end

    new_matrix = y_with_label_vectors;
end

function [inverted] = invert(v)
    inverted = (v .* - 1) + 1;
end

% hx_of_layer3 is [5000 x 10]
hx_of_layer3 = log(layer3);

cost_of_each_example = zeros(size(layer3, 1), 1);
% for each row
% for hx_index = 1:1

for k = 1:size(layer3, 1)
    example_hx_at_k = layer3(k, :);

    % create temp row
    cost_row = zeros(1, size(layer3, 2));

    % example output value

    % when y = 1, cost is (-log(hx))
    y_row = y_with_label_vectors(:, k)';
    y_is_1_cost = -log(example_hx_at_k);
    cost_row += (y_is_1_cost .* y_row);

    % when y = 0, cost is (-log(1-hx))
    inverted_y_row = invert(y_row);
    y_is_0_cost = -log(1 - example_hx_at_k);
    cost_row += (y_is_0_cost .* inverted_y_row);

    cost_sum = sum(cost_row);

    cost_of_each_example(k) = cost_sum;
end

J = (1/size(layer3, 1)) * sum(cost_of_each_example)

%%% QUESTION POINT %%%
%%% QUESTION POINT %%%
%%% QUESTION POINT %%%
% Why did we convert to 10 dim vectors? 

% What do we have now??
% 1. we have the activation layer 3 (the output layer)
%       this is the layer of h(x) values
% 2. we have the output layer (y)

% for every example, we want to do compare the entire 

% cost = sum(layer3 - y_with_label_vectors', 2);

% Cost should be 0.287629


% We need to convert 









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
