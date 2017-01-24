function Tij = calc_Tij(Li, i, j, opt)
% calc_Tij calculates the value of Tij(li) given the specified li
% the origin is located at i th part

% get the observed value
x_i = Li(1);
y_i = Li(2);
theta_i = Li(3);
s_i = Li(4);

% get the relative value and the joint coordinate
% Note the order of i & j when using the relative values
x_ij = opt.model.x_ij(i, j);
y_ij = opt.model.y_ij(i, j);
theta_ij = opt.model.theta_ij(i, j);
s_ij = opt.model.s_ij(i, j);

W_ij = diag([opt.wij.x, opt.wij.y]);
R_theta_i = [cos(theta_i), -sin(theta_i); sin(theta_i), cos(theta_i)];

% compute coordinate for Tij(li)
theta_p_i = opt.wij.theta * (theta_i - theta_ij/2);
s_p_i  = opt.wij.s * (log(s_i) - log(s_ij)/2);
xy_p_i = W_ij * ([x_i, y_i]' + s_i * R_theta_i * [x_ij, y_ij]');
xy_p_i = xy_p_i';

Tij = [xy_p_i, theta_p_i, s_p_i];

end

