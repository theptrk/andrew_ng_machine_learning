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

% Calculate a2
z2 = X * Theta1';
a2 = sigmoid(z2);

% Add bias to a2
a2 = [ones(size(a2, 1), 1), a2];

z3 = a2 * Theta2';
% Calculate a3
a3 = sigmoid(z3);

% y is a vector of labels in range [1, 10] in SOME of the homework
% HOWEVER y can be a vector of labels in ANY range (see argument "num_labels")

% Convert the labels to a "num_labels" dimensional vectors of values in range [0,1] 
% where the index represents the label value. 

% So one is: [1; 0; 0; 0; ... 0]. Two is: [0; 1; 0; 0; ... 0]
y_with_label_vectors = convert_num_labels_dim(y, num_labels);

function [new_matrix] = convert_num_labels_dim(vector_of_num_labels, num_labels)

    % % create an empty "labels as label vectors" container matrix
    labels_as_label_vectors = zeros(num_labels, size(vector_of_num_labels, 1));

    for label_index = 1:size(vector_of_num_labels, 1)
    %     % create a "label vector" container
    %     label_vector = zeros(num_labels, 1);

    %     % get the label value
    %     label_value = vector_of_num_labels(label_index);

    %     % use label_value as the index to set to 1
    %     label_vector(label_value) = 1;

    %     % add "label vector" to container
    %     labels_as_label_vectors(:, label_index) = label_vector;
        labels_as_label_vectors(:, label_index) = (1: num_labels)' == vector_of_num_labels(label_index);
    end

    % % return container matrix
    new_matrix = labels_as_label_vectors;

    % new_matrix = (1: num_labels)' == vector_of_num_labels
end


function [inverted] = invert(v)
    inverted = (v .* - 1) + 1;
end

% intialize placeholder
cost_of_each_example = zeros(size(a3, 1), 1);

% for each row
% for hx_index = 1:1
for k = 1:size(a3, 1)
    example_hx_at_k = a3(k, :);

    % create temp row as placeholder
    cost_row = zeros(1, size(a3, 2));

    % example output value

    % when y = 1, cost is (-log(hx))
    y_row = y_with_label_vectors(:, k)';
    y_is_1_cost = -log(example_hx_at_k);
    cost_row += (y_is_1_cost .* y_row);

    % when y = 0, cost is (-log(1-hx))
    inverted_y_row = invert(y_row);
    y_is_0_cost = -log(1 - example_hx_at_k);
    cost_row += (y_is_0_cost .* inverted_y_row);

    % sum
    cost_sum = sum(cost_row);

    % add to placeholder
    cost_of_each_example(k) = cost_sum;
end

J_without_reg = (1/size(a3, 1)) * sum(cost_of_each_example);

regularization = get_regularization_factor(Theta1, Theta2);

function [regularization] = get_regularization_factor(theta1, theta2)
    % trim the first bias column out of the theta matrix
    Theta1_without_bias = Theta1(:, 2:end);
    Theta2_without_bias = Theta2(:, 2:end);

    Theta1_squared_summed = sum(sum(Theta1_without_bias .* Theta1_without_bias));
    Theta2_sqaured_summed = sum(sum(Theta2_without_bias .* Theta2_without_bias));

    % return regularization
    regularization = (lambda/(2*m)) * (Theta1_squared_summed + Theta2_sqaured_summed);
end

J = J_without_reg + regularization;

% -------------------------------------------------------------

% =========================================================================

% for each example
% for e = 1:size(X, 1)
%     e_input = X(e, :);
    
%     % === step 1: feedforward for z2, a2, z3, a3 ===
%     e_z2 = z2(e, :); % [ 1 x 25 ]
%     % a2 adds a bias factor
%     e_a2 = a2(e, :); % [ 1 x 26 ]
%     e_z3 = z3(e, :); % [ 1 x 10 ]
%     % a3 does not need a bias factor
%     e_a3 = a3(e, :); % [ 1 x 10 ]
%     % label value is converted to label vector
%     e_y = convert_num_labels_dim(y(e), num_labels); % [ 1 x 10 ]
    
%     % === step 2: calculate the error of the output layer ===
%     error_layer3 = e_a3 - e_y';  % [ 1 x 10 ] - [ 1 x 10 ]

%     % === step 3: calculate the error of the hidden layer ===
%     % error_layer2 = theta2 * error_layer3 .* sigmoidGradient(z2)

%     e_sigmoid_gradient = sigmoidGradient(e_z2); % [ 1 x 25 ]
    
%     %  Theta2 is [26 x 10]
%     error_layer2 = (error_layer3 * Theta2)(2:end) .* sigmoidGradient(e_z2);
    
%     % Theta2_grad = e_theta2' * error_layer3 .* sigmoidGradient(e_z2)
%     % additive to theta2_grad is error_layer3 * activation_layer2
%     %                           ([1 x 10] * [1 x 26])
%     % Theta2_grad = Theta2_grad + (error_layer3 * e_a2);
%     Theta2_grad = Theta2_grad + (error_layer3' * e_a2);
%     Theta1_grad = Theta1_grad + (error_layer2' * e_input);
%     % step 4
% end

% refer to page
for i = 1:m
    % feed forward
    a1 = X(i,:)';           % [1 x 401]' = [401 x 1]
    z2 = Theta1 * a1;       % [25 x 401] * [401 x 1] = [25 x 1]
    a2 = [1; sigmoid(z2)];  % [26 x 1]
    z3 = Theta2 * a2;       % [10 x 26] * [26 x 1] = [10 x 1]
    a3 = sigmoid(z3);       % [10 x 1]

    % backprop
    label = y(i);                             % [1]
    label_vector = (1:num_labels)' == label;  % [10 x 1]

    % error layer 3, gradient 2
    error_layer3 = a3 - label_vector;          % [10 x 1] - [10 x 1] = [10 x 1]
    Theta2_grad = Theta2_grad + error_layer3 * a2';
        % Theta2_grad = [10 x 26]
        % error_layer3 = [10 x 1]
        % a2' = [26 x 1]' = [1 x 26]
        % [10 x 26] + ([10 x 1] * [1 x 26]) = 
        % [10 x 26] + ([10 x 26]) = [25 x 26]

    % error layer 2, gradient 1
    error_layer2 = Theta2' * error_layer3 .* (a2 .* (1 - a2));
        % Theta2' = [10 x 26]' = [26 x 10]
        % error_layer3 = [10 x 1]
        % a2 = [26 x 1]
        % [26 x 10] * [10 x 1] .* [26 x 1] = 
        % [26 x 1]             .* [26 x 1] = [26 x 1]
    Theta1_grad = Theta1_grad + error_layer2(2:end) * a1';
end

Theta2_grad = Theta2_grad / m;
Theta1_grad = Theta1_grad / m;

Theta2_without_bias = Theta2(:, 2:end);
Theta1_without_bias = Theta1(:, 2:end);
Theta2_grad_reg = Theta2_without_bias .* (lambda/m);
Theta1_grad_reg = Theta1_without_bias .* (lambda/m);
Theta2_grad = Theta2_grad + [zeros(size(Theta2_grad_reg, 1), 1) Theta2_grad_reg];
Theta1_grad = Theta1_grad + [zeros(size(Theta1_grad_reg, 1), 1) Theta1_grad_reg];

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
