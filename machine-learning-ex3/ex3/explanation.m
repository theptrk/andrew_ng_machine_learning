% %% Initialization
clear; close all; clc

function [nothing] = pprint(str)
	fprintf(str); fprintf('\n');
end
function [nothing] = pprint_run(command)
	fprintf('running >: `%s`\n', command);
end
function [nothing] = ppause(str)
	fprintf('\n'); pause;
end

%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits

%% ==============================================
%% =========== Part X: Loading Data =============
%% ==============================================

% Load Training Data
pprint('Loading and Visualizing Data ...\n')

pprint_run("load('ex3data1.mat')");
load('ex3data1.mat'); % training data stored in arrays X, y

pprint_run('size(X)');
size(X)
pprint("Your data is 5000 examples of 20x20 pixel images with \
	    400 input variables representing the grayscale")
% pprint('400 input variables representing the grayscale')

ppause

pprint('This is the dimensions of one example\n')
pprint_run('size(X(1,:))');
size(X(1,:))

pprint('This is the label for that example (where 10 actually means 0)\n')
pprint_run('y(1)');
y(1)

ppause

pprint('These are all the unique labels in vector y\n')
pprint_run('unique(y)');
unique(y)

ppause; 

pprint('neat huh?'); ppause

clc;

%% ============================================
%% =========== Part X: One vs All =============
%% ============================================
pprint('=== Part X: One vs All ===')

pprint('Note that for One vs All, we want run a separate regression for every label')
ppause()

% Since we are predicting numbers from 1-10, we have 10 labels
num_labels = 10;          % 10 labels, from 1 to 10
                          % (note that we have mapped "0" to label 10)

pprint('Step 1: initialize an empty target theta matrix called "one_vs_all_theta"')
pprint('        this matrix has dimensions: (number of labels) x (number of inputs)')
pprint('        and will store the result of gradient descent for every one vs all')

ppause

% pprint_run('all_theta = zeros(num_labels, size(X, 2) + 1)')
pprint('Note: the (+1) is for the bias theta factor')
% all_theta = zeros(num_labels, size(X, 2) + 1);
% pprint_run('size(all_theta)')
% size(all_theta)

ppause

pprint_run("one_vs_all_theta = zeros(num_labels, size(X, 2) + 1)")
pprint('Note: the (+1) is for the bias theta factor')
one_vs_all_theta = zeros(num_labels, size(X, 2) + 1);

pprint_run('size(one_vs_all_theta)')
size(one_vs_all_theta)

ppause

pprint('Step 2: for loop over every label (c) ')
pprint('Step 2a:   create a one_vs_all_vector (y == c)')
pprint('Step 2b:   run gradient descent => save to "one_vs_all_theta"')

ppause

pprint('For example: if we are iterating on label=4')
pprint('`y == 4` creates a "one_vs_all_vector" for this label:')

ppause

pprint('Note: this octave feature creates our "one_vs_all_vector"')
pprint('> [1;2;3;4;5] == 4')
[1;2;3;4;5] == 4

ppause

pprint_run('X_w_bias = [ones(size(X, 1)) X]')
pprint('Note: the ones vector is for the bias (always 1) input factor')
pprint('X_w_bias should be 5000 * 401')
X_w_bias = [ones(size(X, 1), 1) X];

pprint_run('size(X_w_bias)')
size(X_w_bias)

ppause

for c = 1:num_labels
	lambda = 0.1;
	one_vs_all_vector = y == c;
	initial_theta = zeros(size(X, 2)+1, 1);
	options = optimset('GradObj', 'on', 'MaxIter', 50);

	[theta] = ...
        fmincg (@(t)(lrCostFunction(t, X_w_bias, one_vs_all_vector, lambda)), ...
                initial_theta, options);
    [one_vs_all_theta(c,:)] = theta';
end


%% =========== Part X: One vs All =============
