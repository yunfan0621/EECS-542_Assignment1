clc; close all; clear all;

%% Startup Commands
startup_flag = 1;
if startup_flag
    installmex;
    startup;
end

%% Initialization

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% specify and read in the image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
img_seq = 1;  % the sequence (index) of the image in the folder
img_id  = 63; % the id of the image filename
img_filename  = sprintf('%06d.jpg', img_id);
img_directory = fullfile('..', 'buffy_s5e2_original', img_filename);
img = imread(img_directory);

% Read in the annotations
% lF: 1 x 76 struct: <frame_id, coor 4x6 double>
lF = ReadStickmenAnnotationTxt('../data/buffy_s5e2_sticks.txt');
dat_pt = lF(img_seq).stickmen.coor; % GT endpoint annotation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% specify struct for each part
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
torso = struct('name',        'torso',...
               'part_id',     1,...
               'B',           [],...
               'L',           [],...
               'coor',        []); % root node

head  = struct('name',        'head',...
               'part_id',     6,...
               'B',           [],...
               'L',           [],...
               'coor',        []); % leaf node

upper_arm_r = struct('name',        'upper_arm_r',...
                     'part_id',     3,...
                     'B',           [],...
                     'L',           [],...
                     'coor',        []); % leaf node
                 
upper_arm_l = struct('name',        'upper_arm_l',...
                     'part_id',     2,...
                     'B',           [],...
                     'L',           [],...
                     'coor',        []); % leaf node

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% specify hyper-parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[m, n, ~] = size(img);


%% Compute f(w) for distance transformation and initialize D for leave nodes

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the grid for lj
%%%%%%%%%%%%%%%%%%%%%%%%%%%
lj_x_grid = linspace(1, n, opt.scan_nsample.x);
lj_y_grid = linspace(1, m, opt.scan_nsample.y);
lj_theta_grid = linspace(-pi, pi, opt.scan_nsample.theta); % normalization???
lj_s_grid = linspace(opt.scan_nsample.s_min, opt.scan_nsample.s_max, opt.scan_nsample.s);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initializing D using f(w) for all leave node
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

head.B = zeros(size(torso.B));
upper_arm_r.B = zeros(size(torso.B));
upper_arm_l.B = zeros(size(torso.B));

fprintf('Initializing D using f(w) for all leave nodes...\n');
for x_ind = 1 : length(lj_x_grid)
    
    % display progress
    fprintf('Progress: %.0f%%\n', 100*x_ind/length(lj_x_grid)); 
    
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

%% Compute minimum distance in D for leave nodes

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Forward pass through D to find the minimum value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nForward pass through D for all leave nodes...\n');
for x_ind = 1 : length(lj_x_grid)
    
    % display progress
    if (mod(x_ind,5) == 0) 
        fprintf('Progress: %.0f%%\n', 100*x_ind/length(lj_x_grid)); 
    end
    
    for y_ind = 1 : length(lj_y_grid)
        for theta_ind = 1 : length(lj_theta_grid)
            for s_ind = 1 : length(lj_s_grid)
                
                % initialize neighbor vectors
                head_neighbors = Inf(1, 5);
                upper_arm_r_neighbors = Inf(1, 5);
                upper_arm_l_neighbors = Inf(1, 5);
                
                % retrieve neighboring elements
                head_neighbors(1) = head.B(x_ind, y_ind, theta_ind, s_ind);
                upper_arm_r_neighbors(1) = upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind);
                upper_arm_l_neighbors(1) = upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind);
                
                if (x_ind+1 <= opt.scan_nsample.x) % neighbor in x dimension
                    head_neighbors(2) = head.B(x_ind+1, y_ind, theta_ind, s_ind) + opt.k.x;
                    upper_arm_r_neighbors(2) = upper_arm_r.B(x_ind+1, y_ind, theta_ind, s_ind) + opt.k.x;
                    upper_arm_l_neighbors(2) = upper_arm_l.B(x_ind+1, y_ind, theta_ind, s_ind) + opt.k.x;
                end
                if (y_ind+1 <= opt.scan_nsample.y) % neighbor in y dimension
                    head_neighbors(3) = head.B(x_ind, y_ind+1, theta_ind, s_ind) + opt.k.y;
                    upper_arm_r_neighbors(3) = upper_arm_r.B(x_ind, y_ind+1, theta_ind, s_ind) + opt.k.y;
                    upper_arm_l_neighbors(3) = upper_arm_l.B(x_ind, y_ind+1, theta_ind, s_ind) + opt.k.y;
                end
                if (theta_ind+1 <= opt.scan_nsample.theta) % neighbor in theta dimension
                    head_neighbors(4) = head.B(x_ind, y_ind, theta_ind+1, s_ind) + opt.k.theta;
                    upper_arm_r_neighbors(4) = upper_arm_r.B(x_ind, y_ind, theta_ind+1, s_ind) + opt.k.theta;
                    upper_arm_l_neighbors(4) = upper_arm_l.B(x_ind, y_ind, theta_ind+1, s_ind) + opt.k.theta;
                end
                if (s_ind+1 <= opt.scan_nsample.s) % neighbor in s dimension
                    head_neighbors(5) = head.B(x_ind, y_ind, theta_ind, s_ind+1) + opt.k.s;
                    upper_arm_r_neighbors(5) = upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind+1) + opt.k.s;
                    upper_arm_l_neighbors(5) = upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind+1) + opt.k.s;
                end
                
                % obtain the min value
                head.B(x_ind, y_ind, theta_ind, s_ind) = min(head_neighbors);
                upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind) = min(upper_arm_r_neighbors);
                upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind) = min(upper_arm_l_neighbors);
                
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Backward pass through D to find the minimum value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nBackward pass through D for all leave nodes...\n');
for x_ind = length(lj_x_grid) : -1 : 1
    
    % display progress
    if (mod(x_ind,5) == 0) 
        fprintf('Progress: %.0f%%\n', 100*(length(lj_x_grid)-x_ind+1)/length(lj_x_grid)); 
    end
    
    for y_ind = length(lj_y_grid) : -1 : 1
        for theta_ind = length(lj_theta_grid) : -1 : 1
            for s_ind = length(lj_s_grid) : -1 : 1
                
                % initialize neighbor vectors
                head_neighbors = Inf(1, 5);
                upper_arm_r_neighbors = Inf(1, 5);
                upper_arm_l_neighbors = Inf(1, 5);
                
                % retrieve neighboring elements
                head_neighbors(1) = head.B(x_ind, y_ind, theta_ind, s_ind);
                upper_arm_r_neighbors(1) = upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind);
                upper_arm_l_neighbors(1) = upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind);
                
                if (x_ind-1 >= 1) % neighbor in x dimension
                    head_neighbors(2) = head.B(x_ind-1, y_ind, theta_ind, s_ind) + opt.k.x;
                    upper_arm_r_neighbors(2) = upper_arm_r.B(x_ind-1, y_ind, theta_ind, s_ind) + opt.k.x;
                    upper_arm_l_neighbors(2) = upper_arm_l.B(x_ind-1, y_ind, theta_ind, s_ind) + opt.k.x;
                end
                if (y_ind-1 >= 1) % neighbor in y dimension
                    head_neighbors(3) = head.B(x_ind, y_ind-1, theta_ind, s_ind) + opt.k.y;
                    upper_arm_r_neighbors(3) = upper_arm_r.B(x_ind, y_ind-1, theta_ind, s_ind) + opt.k.y;
                    upper_arm_l_neighbors(3) = upper_arm_l.B(x_ind, y_ind-1, theta_ind, s_ind) + opt.k.y;
                end
                if (theta_ind-1 >= 1) % neighbor in theta dimension
                    head_neighbors(4) = head.B(x_ind, y_ind, theta_ind-1, s_ind) + opt.k.theta;
                    upper_arm_r_neighbors(4) = upper_arm_r.B(x_ind, y_ind, theta_ind-1, s_ind) + opt.k.theta;
                    upper_arm_l_neighbors(4) = upper_arm_l.B(x_ind, y_ind, theta_ind-1, s_ind) + opt.k.theta;
                end
                if (s_ind-1 >= 1) % neighbor in s dimension
                    head_neighbors(5) = head.B(x_ind, y_ind, theta_ind, s_ind-1) + opt.k.s;
                    upper_arm_r_neighbors(5) = upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind-1) + opt.k.s;
                    upper_arm_l_neighbors(5) = upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind-1) + opt.k.s;
                end
                
                % obtain the min value
                head.B(x_ind, y_ind, theta_ind, s_ind) = min(head_neighbors);
                upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind) = min(upper_arm_r_neighbors);
                upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind) = min(upper_arm_l_neighbors);
                
            end
        end
    end
end

%% Compute f(w) for distance transformation and initialize D for root node

torso.B = zeros(opt.scan_nsample.x, opt.scan_nsample.y, opt.scan_nsample.theta, opt.scan_nsample.s);

fprintf('Initializing D using f(w) for all leave nodes...\n');
for x_ind = 1 : length(lj_x_grid)
    
    % display progress
    fprintf('Progress: %.0f%%\n', 100*x_ind/length(lj_x_grid)); 
    
    for y_ind = 1 : length(lj_y_grid)
        for theta_ind = 1 : length(lj_theta_grid)
            for s_ind = 1 : length(lj_s_grid)
                
                % retrieve the coordinate and energy of leave nodes
                x = lj_x_grid(x_ind);
                y = lj_y_grid(y_ind);
                s = lj_s_grid(s_ind);
                theta = lj_theta_grid(theta_ind);
                
                L = [x, y, theta, s];
                head_energy = head.B(x_ind, y_ind, theta_ind, s_ind);
                upper_arm_r_energy = upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind);
                upper_arm_l_energy = upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind);
                
                % compute the total energy
                torso.B(x_ind, y_ind, theta_ind, s_ind) = ...
                    match_energy_cost(L, torso.part_id, dat_pt(:,torso.part_id)) + ...
                    head_energy + upper_arm_r_energy + upper_arm_r_energy;  
            end
        end
    end
end

%% Compute minimum distance in D for root node

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Forward pass through D to find the minimum value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nForward pass through D for the root node...\n');

% keep track of the lowest energy and the associated configuration
torso_opt_L = [1, 1, 1, 1];
torso_opt_E = Inf;
for x_ind = 1 : length(lj_x_grid)
    
    % display progress
    if (mod(x_ind,5) == 0) 
        fprintf('Progress: %.0f%%\n', 100*x_ind/length(lj_x_grid)); 
    end
    
    for y_ind = 1 : length(lj_y_grid)
        for theta_ind = 1 : length(lj_theta_grid)
            for s_ind = 1 : length(lj_s_grid)
                
                % initialize neighbor vectors
                neighbors = Inf(1, 5);
                
                % retrieve neighboring elements
                neighbors(1) = head.B(x_ind, y_ind, theta_ind, s_ind);
                
                if (x_ind+1 <= opt.scan_nsample.x) % neighbor in x dimension
                    neighbors(2) = torso.B(x_ind+1, y_ind, theta_ind, s_ind) + opt.k.x;
                end
                if (y_ind+1 <= opt.scan_nsample.y) % neighbor in y dimension
                    neighbors(3) = torso.B(x_ind, y_ind+1, theta_ind, s_ind) + opt.k.y;
                end
                if (theta_ind+1 <= opt.scan_nsample.theta) % neighbor in theta dimension
                    neighbors(4) = torso.B(x_ind, y_ind, theta_ind+1, s_ind) + opt.k.theta;
                end
                if (s_ind+1 <= opt.scan_nsample.s) % neighbor in s dimension
                    neighbors(5) = torso.B(x_ind, y_ind, theta_ind, s_ind+1) + opt.k.s;
                end
                
                % obtain the min value
                torso.B(x_ind, y_ind, theta_ind, s_ind) = min(neighbors);
                
                % update the min value and coordinate
                if (torso.B(x_ind, y_ind, theta_ind, s_ind) < torso_opt_E)
                    torso_opt_L = [x_ind, y_ind, theta_ind, s_ind];
                    torso_opt_E = torso.B(x_ind, y_ind, theta_ind, s_ind);
                end
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Backward pass through D to find the minimum value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% do not initialize minimum value and coordinate at this time
fprintf('\nBackward pass through D for all leave nodes...\n');
for x_ind = length(lj_x_grid) : -1 : 1
    
    % display progress
    if (mod(x_ind,5) == 0) 
        fprintf('Progress: %.0f%%\n', 100*(length(lj_x_grid)-x_ind+1)/length(lj_x_grid)); 
    end
    
    for y_ind = length(lj_y_grid) : -1 : 1
        for theta_ind = length(lj_theta_grid) : -1 : 1
            for s_ind = length(lj_s_grid) : -1 : 1
                
                % initialize neighbor vectors
                neighbors = Inf(1, 5);
                
                % retrieve neighboring elements
                neighbors(1) = head.B(x_ind, y_ind, theta_ind, s_ind);
                
                if (x_ind-1 >= 1) % neighbor in x dimension
                    neighbors(2) = torso.B(x_ind-1, y_ind, theta_ind, s_ind) + opt.k.x;
                end
                if (y_ind-1 >= 1) % neighbor in y dimension
                    neighbors(3) = torso.B(x_ind, y_ind-1, theta_ind, s_ind) + opt.k.y;
                end
                if (theta_ind-1 >= 1) % neighbor in theta dimension
                    neighbors(4) = torso.B(x_ind, y_ind, theta_ind-1, s_ind) + opt.k.theta;
                end
                if (s_ind-1 >= 1) % neighbor in s dimension
                    neighbors(5) = torso.B(x_ind, y_ind, theta_ind, s_ind-1) + opt.k.s;
                end
                
                % obtain the min value
                torso.B(x_ind, y_ind, theta_ind, s_ind) = min(neighbors);
                
                % update the min value and coordinate
                if (torso.B(x_ind, y_ind, theta_ind, s_ind) < torso_opt_E)
                    torso_opt_L = [x_ind, y_ind, theta_ind, s_ind];
                    torso_opt_E = torso.B(x_ind, y_ind, theta_ind, s_ind);
                end
                
            end
        end
    end
end

%% Obtain the coordinate of each node
% torso
torso_x = lj_x_grid(torso_opt_L(1));
torso_y = lj_y_grid(torso_opt_L(2));
torso_theta = lj_theta_grid(torso_opt_L(3));
torso_s = lj_s_grid(torso_opt_L(4));



