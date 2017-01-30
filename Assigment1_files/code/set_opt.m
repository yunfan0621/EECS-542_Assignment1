%% Create parameter struct opt 
% specify L grid
opt.scan_nsample.x = 50;
opt.scan_nsample.y = 50;
opt.scan_nsample.theta = 20;
opt.scan_nsample.theta_max = pi;
opt.scan_nsample.theta_min = -pi;
opt.scan_nsample.s = 10;
opt.scan_nsample.s_min = 0.2;
opt.scan_nsample.s_max = 2;

% compensate constant value for D pass through 
opt.k = [];
opt.k.x = 0;
opt.k.y = 0;
opt.k.theta = 0;
opt.k.s = 0;

% % % weight for adding fw
% % opt.fw.weight = 1;

% weights for computing dij
opt.wij.weight = 1; % weight for deformation term
opt.wij.x = opt.wij.weight * 0.15;
opt.wij.y = opt.wij.weight * 0.15;
opt.wij.theta = opt.wij.weight * 100;
opt.wij.s = opt.wij.weight * 1000;

% parameter of the ideal model
opt.model.len = [160, 95, 95, 65, 65, 60]; % length of model part measured in pixels
                                           % [torso, upper_arm_r, upper_arm_l, lower_arm_r, lower_arm_l, head]
opt.model.x_ij = zeros(length(opt.model.len));
opt.model.y_ij = zeros(length(opt.model.len));
opt.model.theta_ij = zeros(length(opt.model.len));
opt.model.s_ij = zeros(length(opt.model.len));

%% specify coordinate of joints
% NOTE the direction of x-y axis

% torso (li) -> upper_arm_r (lj)
opt.model.x_ij(torso.part_id, upper_arm_r.part_id) = 60;
opt.model.y_ij(torso.part_id, upper_arm_r.part_id) = -60;
opt.model.theta_ij(torso.part_id, upper_arm_r.part_id) = 0;
opt.model.s_ij(torso.part_id, upper_arm_r.part_id) = 1;

% torso (li) -> upper_arm_l (lj)
opt.model.x_ij(torso.part_id, upper_arm_l.part_id) = -60;
opt.model.y_ij(torso.part_id, upper_arm_l.part_id) = -60;
opt.model.theta_ij(torso.part_id, upper_arm_l.part_id) = 0;
opt.model.s_ij(torso.part_id, upper_arm_l.part_id) = 1;

% torso (li) -> head (lj)
opt.model.x_ij(torso.part_id, head.part_id) = 0;
opt.model.y_ij(torso.part_id, head.part_id) = -87.5;
opt.model.theta_ij(torso.part_id, head.part_id) = 0;
opt.model.s_ij(torso.part_id, head.part_id) = 1;

% upper_arm_r (li) -> torso (lj)
opt.model.x_ij(upper_arm_r.part_id, torso.part_id) = -47.5;
opt.model.y_ij(upper_arm_r.part_id, torso.part_id) = 0;
opt.model.theta_ij(upper_arm_r.part_id, torso.part_id) = 0;
opt.model.s_ij(upper_arm_r.part_id, torso.part_id) = 1;

% upper_arm_l (li) -> torso (lj)
opt.model.x_ij(upper_arm_l.part_id, torso.part_id) = 47.5;
opt.model.y_ij(upper_arm_l.part_id, torso.part_id) = 0;
opt.model.theta_ij(upper_arm_l.part_id, torso.part_id) = 0;
opt.model.s_ij(upper_arm_l.part_id, torso.part_id) = 1;

% head (li) -> torso (lj)
opt.model.x_ij(head.part_id, torso.part_id) = 0;
opt.model.y_ij(head.part_id, torso.part_id) = 37.5;
opt.model.theta_ij(head.part_id, torso.part_id) = 0;
opt.model.s_ij(head.part_id, torso.part_id) = 1;