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

a_1 = 7;
a_2 = 2;

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


%backpropagation unregularized code below.


temp = (1 - Output_out).*(1 + Output_out);
del_3 = a_2*(Output_out - y) .* temp;    % error per output node for all m training sets for output layer 

weighted_avg_of_3_for_2 = del_3*Theta2;                       % gives a 5000*26 matrix. ( Theta2 includes weight for bias node.)

% del_2 is a 5000*25 (m*num_hidden_nodes_excluding_bias) matrix as no del for bias.

del_2 = a_1*weighted_avg_of_3_for_2(:,2:end) .* sigmoidGradient(Hidden_in);  % we don't have to calculate for the bias node. It's only for feed forward.

Hidden_out_for_grad = [zeros(m,1) Hidden_out(:,2:end)];

Theta2_grad = Theta2_grad + (del_3'*Hidden_out);

Theta2_grad = Theta2_grad/m;

X_for_grad = [zeros(m,1) X(:,2:end)];

Theta1_grad = Theta1_grad + (del_2'*X);

Theta1_grad = Theta1_grad/m;

%disp(Theta1_grad);
%disp("----------");
%disp(Theta2_grad);

%regularising both theta_grad.

Theta1_for_reg = [zeros(size(Theta1,1),1) Theta1(:,2:end)];

Theta2_for_reg = [zeros(size(Theta2,1),1) Theta2(:,2:end)];

Theta1_grad = Theta1_grad + lambda*Theta1_for_reg/m; 

Theta2_grad = Theta2_grad + lambda*Theta2_for_reg/m; 









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
