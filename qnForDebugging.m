function qnForDebugging
x = [-1.2; 1.0]
N = numel(x);
B = eye(N);
[fval,grad] = objective(x);
k = 0;
%-------------------
disp('---------')
k, x', fval
disp('---------')
%-------------------
while k <= 100
  k = k + 1;
  dir = B \ (-grad);
  fOld = fval;
  gradOld = grad;
  % Backtracking
  alpha = 2.0;
  while fval >= fOld
    alpha = alpha/2.0;
    fval = objective(x + alpha*dir)
  end
  deltaX = alpha*dir;
  x = x + deltaX;
  [fval,grad] = objective(x);
  deltaF = fval - fOld;
  deltaGrad = grad - gradOld;
  BdeltaX = B*deltaX;
  B = B + (deltaGrad*deltaGrad')/(deltaGrad'*deltaX) - BdeltaX*BdeltaX'/(deltaX'*BdeltaX);
  %-------------------
  disp('---------')
  k, x', fval, alpha
  disp('---------')
  %-------------------
end
end
function [fval,grad] = objective(x)
% Calculate objective f
fval = 100*(x(2) - x(1)^2)^2 + (1-x(1))^2;
if nargout > 1 % gradient required
    grad = [-400*(x(2)-x(1)^2)*x(1)-2*(1-x(1)); 200*(x(2)-x(1)^2)];
end
%  fval = 0.5*(x(1)*x(1) + 10.0*x(2)*x(2));
%  grad = [x(1); 10.0*x(2)];
end
