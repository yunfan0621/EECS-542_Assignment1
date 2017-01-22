function Tij = calc_Tij(Li, i, j, opt)
% calc_Tij calculates the value of Tij(li) given the specified li
% the origin is located at i th part

% get the observed value
x_i = Li(1);
y_i = Li(2);
theta_i = Li(3);
s_i = Li(4);

% get the relative value
theta_ij = opt.model.theta_ij(i, j);
s_ij = opt.model.s_ij(i, j);
W_ij = diag(opt.wij.x, opt.wij.y);
R_theta_i = [cos(theta_i), -sin(theta_i); sin(theta_i), cos(theta_i)];

end

