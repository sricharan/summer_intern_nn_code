function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

a_1 = 7;

g = 1.0 ./ (1.0 + exp(-a_1*z));
end
