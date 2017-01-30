function Tij = calc_Tij(i, j, x_grid, y_grid, theta_grid, s_grid, opt)

% compute theta_i' and s_i' for computing Tij_xy
Tij_theta = opt.wij.theta * (theta_grid - opt.model.theta_ij(i, j)/2);
Tij_s     = opt.wij.s * (log(s_grid) - log(opt.model.s_ij(i, j)));

% compute theta' and s' for Tij
vec_theta_s = combvec(theta_grid, s_grid); % All combinations of (theta', s')
R_theta_s = zeros(2*length(vec_theta_s));  % a large matrix storing the rotation matrices for combinations of (theta, s)
for k = 1 : length(vec_theta_s)
    % for every combination of (theta, s)
    theta_tmp = vec_theta_s(1, k);
    s_tmp     = vec_theta_s(2, k);
    R_theta_i_tmp = s_tmp * [cos(theta_tmp), -sin(theta_tmp); sin(theta_tmp), cos(theta_tmp)];
    R_theta_s((2*k-1):2*k, (2*k-1):2*k) = R_theta_i_tmp;
end

% coordinate of joint with each possible (theta, s) in the frame of part
joint_xy_frame = repmat([opt.model.x_ij(i, j); opt.model.y_ij(i, j)], length(vec_theta_s), 1);
joint_xy_frame_rotate = R_theta_s * joint_xy_frame;
joint_xy_frame_rotate = [joint_xy_frame_rotate(1:2:end)'; joint_xy_frame_rotate(2:2:end)']; % coordinate of joint with each possible (theta, s)

% coordinate of joint with each possible (theta, s) in the world frame
center_xy_world = combvec(x_grid, y_grid);
W_ij = diag([opt.wij.x, opt.wij.y]);
Tij_xy = combvec(center_xy_world, joint_xy_frame_rotate); % note the order
Tij_xy = W_ij * [1,0,1,0; 0,1,0,1] * Tij_xy; % add up the center point proposals and the rotated and scaled joints

% obtain the final combination of theta and s
Tij_theta_s = combvec(x_grid, y_grid, Tij_theta, Tij_s);
Tij = [Tij_xy; Tij_theta_s(3:4,:)];

end

