input_layer_size = 2;
hidden_layer_size = 25;
num_labels = 5;

nn_params = load("nn_params.txt");


% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
%[nn_params, cost] = trainNN( X, y, 1);

lambda = 0;

testNN(nn_params, input_layer_size, hidden_layer_size, num_labels, lambda);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));




