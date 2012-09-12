function [lambda_vec, error_train, error_val] = validationCurve(input_layer_size, hidden_layer_size, num_labels, X, y, Xval, yval)

% Number of training examples
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';


% You need to return these values correctly
error_train = zeros(length(lambda_vec), 1);
error_val   = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------


for i = 1:length(lambda_vec)
  
  disp(i);
  all_theta = trainNN(X, y, lambda_vec(i));
  error_train(i) = nnCostFunction(all_theta, input_layer_size, hidden_layer_size, num_labels,X, y, lambda_vec(i));
  error_val(i) = nnCostFunction(all_theta, input_layer_size, hidden_layer_size, num_labels, Xval, yval, lambda_vec(i)); 




% -------------------------------------------------------------

% =========================================================================

end

A = [error_train error_val];

save("-ascii", "validation_curve_data.txt", "A");
