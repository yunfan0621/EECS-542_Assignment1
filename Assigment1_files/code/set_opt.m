% specify L grid
opt.scan_nsample = [];
opt.scan_nsample.x = 50;
opt.scan_nsample.y = 50;
opt.scan_nsample.theta = 20;
opt.scan_nsample.s = 20;
opt.scan_nsample.s_min = 0.1;
opt.scan_nsample.s_max = 20;

% compensate constant value for D pass through 
opt.k = [];
opt.k.x = 0;
opt.k.y = 0;
opt.k.theta = 0;
opt.k.s = 0;

% weights for computing dij
opt.wij.x = 0.5;
opt.wij.y = 0.5;
opt.wij.theta = 0.5;
opt.wij.s = 100;

% parameter of the ideal model
opt.model.len = [160, 95, 95, 65, 65, 60]; % length of model part measured in pixels
                                           % [torso, upper_arm_r, upper_arm_l, lower_arm_r, lower_arm_l, head]
opt.model.x_ij = cell(length(opt.model.len));
opt.model.y_ij = cell(length(opt.model.len));
opt.model.theta_ij = cell(length(opt.model.len));
opt.model.s_ij = cell(length(opt.model.len));