clc; close all;

%% Startup Commands
startup_flag = 0;
if startup_flag
    installmex;
    startup;
end

%% Initialization

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% specify and read in the image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
img_seq = 1;  % the sequence of the image in the folder
img_id  = 63; % the id of the image filename
img_filename  = sprintf('%06d.jpg', img_id);
img_directory = fullfile('..', 'buffy_s5e2_original', img_filename);
img = imread(img_directory);
[m, n, ~] = size(img);

% Read in the annotations
% lF: 1 x 76 struct: <frame_id, coor 4x6 double>
lF = ReadStickmenAnnotationTxt('../data/buffy_s5e2_sticks.txt');
dat_pt = lF(img_seq).stickmen.coor;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% specify hyper-parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opt.scan_nsample = []; % step size for (x,y,theta,s) search
opt.scan_nsample.x = 40;
opt.scan_nsample.y = 40;
opt.scan_nsample.theta = 20;
opt.scan_nsample.s = 30;

opt.model_len = [160, 95, 95, 65, 65, 60]; % length of model part measured in pixels
                                           % [torso, upper_arm_r, upper_arm_l, lower_arm_r, lower_arm_l, head]
                                           
% TODO: understand and specify kx. ky, ktheta and ks

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% specify struct for each part
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
torso = struct('name',        'torso',...
               'id',          1,...
               'part_id',     1,...
               'parent_id',   -1,...
               'children_id', [2, 3, 4],...
               'B',          []); % root node

head  = struct('name',        'head',...
               'id',          2,...
               'part_id',     6,...
               'parent_id',   1,...
               'children_id', [],...
               'B',          []); % leaf node

upper_arm_r = struct('name',        'upper_arm_r',...
                     'id',          3,...
                     'part_id',     3,...
                     'parent_id',   1,...
                     'children_id', [],...
                     'B',          []); % leaf node
                 
upper_arm_l = struct('name',        'upper_arm_l',...
                     'id',          4,...
                     'part_id',     2,...
                     'parent_id',   1,...
                     'children_id', [],...
                     'B',          []); % leaf node
T = [torso, head, upper_arm_r, upper_arm_l];

%% Compute f(w) for distance transformation

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the grid for lj
%%%%%%%%%%%%%%%%%%%%%%%%%%%
lj_x_grid = linspace(1, n, opt.scan_nsample.x);
lj_y_grid = linspace(1, m, opt.scan_nsample.y);
lj_theta_grid = linspace(-pi/2, pi/2, opt.scan_nsample.theta); % inclusive???
lj_s_grid = linspace(0.5, 2, opt.scan_nsample.s); % reasonable???

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initializing D using f(w) for all leave node
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

torso.B = zeros(opt.scan_nsample.x, opt.scan_nsample.y, opt.scan_nsample.theta, opt.scan_nsample.s);
head.B = zeros(size(torso.B));
upper_arm_r.B = zeros(size(torso.B));
upper_arm_l.B = zeros(size(torso.B));

fprintf('Initializing D using f(w) for all leave nodes\n');
for x_ind = 1 : length(lj_x_grid)
    fprintf('Progress: %d/%d...\n', x_ind, length(lj_x_grid));
    for y_ind = 1 : length(lj_y_grid)
        for theta_ind = 1 : length(lj_theta_grid)
            for s_ind = 1 : length(lj_s_grid)
                
                % retrieve the coordinate
                x = lj_x_grid(x_ind);
                y = lj_y_grid(y_ind);
                s = lj_s_grid(s_ind);
                theta = lj_theta_grid(theta_ind);
                
                L = [x, y, theta, s];
                
                % compute the matching cost
                head.B(x_ind, y_ind, theta_ind, s_ind) = match_energy_cost(L, head.part_id, dat_pt(:,head.part_id));
                upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind) = match_energy_cost(L, upper_arm_r.part_id, dat_pt(:,upper_arm_r.part_id));
                upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind) = match_energy_cost(L, upper_arm_l.part_id, dat_pt(:,upper_arm_l.part_id));                
            end
        end
    end
end
