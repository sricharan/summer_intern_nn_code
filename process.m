input_layer_size = 2;
hidden_layer_size = 11;
num_labels = 5;

data = load("training_data.txt");

val_data = load("validation_data.txt");

X = data(:,[1,2]);

y = data(:,[3,4,5,6,7]);

Xval = val_data(:,[1,2]);

yval = val_data(:,[3,4,5,6,7]);

X = featureNormalize(X);

y = featureNormalize(y);

Xval = featureNormalize(Xval);

yval = featureNormalize(yval);


% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = trainNN( X, y, 1);

%% =========== Part 5: Learning Curve for Linear Regression =============
%  Next, you should implement the learningCurve function. 
%
%  Write Up Note: Since the model is underfitting the data, we expect to
%                 see a graph with "high bias" -- slide 8 in ML-advice.pdf 
%

lambda = 0;
[error_train, error_val] = ...
    learningCurve( input_layer_size, hidden_layer_size, num_labels, X, y, ...
                   Xval, yval, ...
                  lambda);
m = size(error_train,1);

plot(1:m, error_train, 1:m, error_val);
title('Learning curve for neural network')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
%axis([0 13 0 150])


fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end






% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


%testNN(Theta1, Theta2);


