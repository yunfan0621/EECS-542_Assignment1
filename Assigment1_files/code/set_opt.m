%% Create parameter struct opt 
% specify L grid
opt.scan_nsample = [];
opt.scan_nsample.x = 50;
opt.scan_nsample.y = 50;
opt.scan_nsample.theta = 20;
opt.scan_nsample.s = 20;
opt.scan_nsample.s_min = 0.1;
opt.scan_nsample.s_max = 2;

% compensate constant value for D pass through 
opt.k = [];
opt.k.x = 0;
opt.k.y = 0;
opt.k.theta = 0;
opt.k.s = 0;

% weights for computing dij
opt.wij.x = 10;
opt.wij.y = 10;
opt.wij.theta = 0.5;
opt.wij.s = 50;

% parameter of the ideal model
opt.model.len = [160, 95, 95, 65, 65, 60]; % length of model part measured in pixels
                                           % [torso, upper_arm_r, upper_arm_l, lower_arm_r, lower_arm_l, head]
opt.model.x_ij = NaN(length(opt.model.len));
opt.model.y_ij = NaN(length(opt.model.len));
opt.model.theta_ij = NaN(length(opt.model.len));
opt.model.s_ij = NaN(length(opt.model.len));

%% specify coordinate of joints
% NOTE the direction of x-y axis

% torso (li) -> upper_arm_r (lj)
opt.model.x_ij(torso.part_id, upper_arm_r.part_id) = 0;
opt.model.y_ij(torso.part_id, upper_arm_r.part_id) = -60;
opt.model.theta_ij(torso.part_id, upper_arm_r.part_id) = -pi/2;
% opt.model.s_ij(torso.part_id, upper_arm_r.part_id) = opt.model.len(torso.part_id) / opt.model.len(upper_arm_r.part_id);
opt.model.s_ij(torso.part_id, upper_arm_r.part_id) = 1;

% torso (li) -> upper_arm_l (lj)
opt.model.x_ij(torso.part_id, upper_arm_l.part_id) = 0;
opt.model.y_ij(torso.part_id, upper_arm_l.part_id) = -60;
opt.model.theta_ij(torso.part_id, upper_arm_l.part_id) = pi/2;
% opt.model.s_ij(torso.part_id, upper_arm_l.part_id) = opt.model.len(torso.part_id) / opt.model.len(upper_arm_l.part_id);
opt.model.s_ij(torso.part_id, upper_arm_l.part_id) = 1;

% torso (li) -> head (lj)
opt.model.x_ij(torso.part_id, head.part_id) = 0;
opt.model.y_ij(torso.part_id, head.part_id) = -87.5;
opt.model.theta_ij(torso.part_id, head.part_id) = 0;
% opt.model.s_ij(torso.part_id, head.part_id) = opt.model.len(torso.part_id) / opt.model.len(head.part_id); % ???
opt.model.s_ij(torso.part_id, head.part_id) = 1;

% upper_arm_r (li) -> torso (lj)
opt.model.x_ij(upper_arm_r.part_id, torso.part_id) = -47.5;
opt.model.y_ij(upper_arm_r.part_id, torso.part_id) = 0;
opt.model.theta_ij(upper_arm_r.part_id, torso.part_id) = pi/2;
% opt.model.s_ij(upper_arm_r.part_id, torso.part_id) = opt.model.len(upper_arm_r.part_id) / opt.model.len(torso.part_id);
opt.model.s_ij(upper_arm_r.part_id, torso.part_id) = 1;

% upper_arm_l (li) -> torso (lj)
opt.model.x_ij(upper_arm_l.part_id, torso.part_id) = 47.5;
opt.model.y_ij(upper_arm_l.part_id, torso.part_id) = 0;
opt.model.theta_ij(upper_arm_l.part_id, torso.part_id) = -pi/2;
% opt.model.s_ij(upper_arm_l.part_id, torso.part_id) = opt.model.len(upper_arm_l.part_id) / opt.model.len(torso.part_id);
opt.model.s_ij(upper_arm_l.part_id, torso.part_id) = 1;

% head (li) -> torso (lj)
opt.model.x_ij(head.part_id, torso.part_id) = 0;
opt.model.y_ij(head.part_id, torso.part_id) = 37.5;
opt.model.theta_ij(head.part_id, torso.part_id) = 0;
% opt.model.s_ij(head.part_id, torso.part_id) = opt.model.len(head.part_id) / opt.model.len(torso.part_id);
opt.model.s_ij(head.part_id, torso.part_id) = 1;