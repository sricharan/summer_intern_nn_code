input_layer_size = 2;
hidden_layer_size = 25;
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
%[nn_params, cost] = trainNN( X, y, 1);

%validation_curve

[lambda_vec, error_train, error_val] = ...
    validationCurve(input_layer_size,hidden_layer_size, num_labels, X, y, Xval, yval);

close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end







% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


%testNN(Theta1, Theta2);


