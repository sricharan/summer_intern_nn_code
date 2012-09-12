function J = feedforward(nn_params, ...
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
%disp(X);         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m,1) X];
Hidden_in = X*Theta1';
Hidden_out = sigmoid(X*Theta1');
%disp(Hidden_out);
Hidden_out = [ones(m,1) Hidden_out];    %adding bias column
Output_in = Hidden_out*Theta2';
Output_out = tansigmoid(Hidden_out*Theta2');
[max_predictions,p] = max(Output_out,[],2);
cost = (y - Output_out).^2;      % TO BE CHANGED
cost_sum = sum(cost,2);

J = sum(cost_sum)/(2*m);

% regularised cost function code below.

Theta1_for_reg_sum = Theta1(:,(2:end)).^2;
Theta1_for_reg_sum = [zeros(size(Theta1,1),1) Theta1_for_reg_sum];
sum_Theta1 = sum(Theta1_for_reg_sum,2);     %summation along rows
sum_Theta1 = sum(sum_Theta1);   %summation along column ( after sum along rows )


Theta2_for_reg_sum = Theta2(:,(2:end)).^2;
Theta2_for_reg_sum = [zeros(size(Theta2,1),1) Theta2_for_reg_sum];
sum_Theta2 = sum(Theta2_for_reg_sum,2);     %summation along rows
sum_Theta2 = sum(sum_Theta2);   %summation along column ( after sum along rows )

reg_term = lambda*(sum_Theta1 + sum_Theta2)/(2*m);

J = J + reg_term;
