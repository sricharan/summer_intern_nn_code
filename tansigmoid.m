function g = tansigmoid(z)
%TANSIGMOID Compute sigmoid functoon
%   J = TANSIGMOID(z) computes the sigmoid of z.

a_2 = 2;

z = a_2*z;

g = (exp(z) - exp(-z)) ./ (exp(z) + exp(-z));
end
