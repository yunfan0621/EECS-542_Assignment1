clc; close all; clear all;

%% Initialization

% Startup command
installmex;
startup;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% specify and read in the image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
img_seq = 35;  % the sequence (index) of the image in the folder
img_id  = 7526; % the id of the image filename
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
               'Bj_p',        [],...
               'coor',        []); % root node

head  = struct('name',        'head',...
               'part_id',     6,...
               'B',           [],...
               'Bj_p',        [],...
               'coor',        []); % leaf node

upper_arm_r = struct('name',        'upper_arm_r',...
                     'part_id',     3,...
                     'B',           [],...
                     'Bj_p',        [],...
                     'coor',        []); % leaf node
                 
upper_arm_l = struct('name',        'upper_arm_l',...
                     'part_id',     2,...
                     'B',           [],...
                     'Bj_p',        [],...
                     'coor',        []); % leaf node

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% specify hyper-parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
set_opt; % Create parameter struct opt

[m, n, ~] = size(img);
x_grid = linspace(1, n, opt.scan_nsample.x);
y_grid = linspace(1, m, opt.scan_nsample.y);
theta_grid = linspace(-pi, pi, opt.scan_nsample.theta); % normalization???
s_grid = linspace(opt.scan_nsample.s_min, opt.scan_nsample.s_max, opt.scan_nsample.s);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% assign spaces for N-D arrays
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% B(li) = min(D(Tij(li))), the minimum distance indexed by li
torso.B = zeros(opt.scan_nsample.x, opt.scan_nsample.y, opt.scan_nsample.theta, opt.scan_nsample.s);
head.B  = zeros(size(torso.B));
upper_arm_r.B = zeros(size(torso.B));
upper_arm_l.B = zeros(size(torso.B));

% Bj_p(li) = argmin(D(Tij(li))), the lj associated with the minimum
% distance (propagated)
torso.Bj_p = cell(opt.scan_nsample.x, opt.scan_nsample.y, opt.scan_nsample.theta, opt.scan_nsample.s);
head.Bj_p  = cell(size(torso.B));
upper_arm_r.Bj_p = cell(size(torso.B));
upper_arm_l.Bj_p = cell(size(torso.B));

%% Compute f(w) for distance transformation and initialize D for leave nodes

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initializing D using f(w) for all leave node
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Initializing D using f(w) for all leave nodes...\n');
for x_ind = 1 : length(x_grid)
    
    % display progress
    fprintf('Progress: %.0f%%\n', 100*x_ind/length(x_grid)); 
    
    for y_ind = 1 : length(y_grid)
        for theta_ind = 1 : length(theta_grid)
            for s_ind = 1 : length(s_grid)
                
                % retrieve the coordinate
                x = x_grid(x_ind);
                y = y_grid(y_ind);
                s = s_grid(s_ind);
                theta = theta_grid(theta_ind);
                
                L = [x, y, theta, s];
                
                % compute the matching cost
                head.B(x_ind, y_ind, theta_ind, s_ind) = match_energy_cost(L, head.part_id, dat_pt(:,head.part_id));
                upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind) = match_energy_cost(L, upper_arm_r.part_id, dat_pt(:,upper_arm_r.part_id));
                upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind) = match_energy_cost(L, upper_arm_l.part_id, dat_pt(:,upper_arm_l.part_id));                
            
                % initialize the Bj_p
                head.Bj_p{x_ind, y_ind, theta_ind, s_ind} = L;
                upper_arm_r.Bj_p{x_ind, y_ind, theta_ind, s_ind} = L;
                upper_arm_l.Bj_p{x_ind, y_ind, theta_ind, s_ind} = L;
            end
        end
    end
end

%% Compute minimum distance in D for leave nodes

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Forward pass through D to find the minimum value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nForward pass through D for all leave nodes...\n');
for x_ind = 1 : length(x_grid)
    
    % display progress
    if (mod(x_ind,5) == 0) 
        fprintf('Progress: %.0f%%\n', 100*x_ind/length(x_grid)); 
    end
    
    for y_ind = 1 : length(y_grid)
        for theta_ind = 1 : length(theta_grid)
            for s_ind = 1 : length(s_grid)
                
                % initialize neighbor vectors
                head_neighbors = Inf(1, 5); % for min value
                upper_arm_r_neighbors = Inf(1, 5);
                upper_arm_l_neighbors = Inf(1, 5);
                
                head_neighbors_lj = zeros(5, 4); % for argmin value, 5-neighbors
                upper_arm_r_neighbors_lj = zeros(5, 4);
                upper_arm_l_neighbors_lj = zeros(5, 4);
                
                % retrieve the coordinate
                x = x_grid(x_ind);
                y = y_grid(y_ind);
                s = s_grid(s_ind);
                theta = theta_grid(theta_ind);
                
                Li = [x, y, theta, s]; % Li
                
                % retrieve neighboring elements
                
                % include itself as well
                Lj = Li;
                Tij = calc_Tij(Li, torso.part_id, head.part_id,  opt);
                Tji = calc_Tij(Lj, head.part_id,  torso.part_id, opt);
                dij = sum(abs(Tij - Tji));
                head_neighbors(1) = dij + head.B(x_ind, y_ind, theta_ind, s_ind);
                
                Tij = calc_Tij(Li, torso.part_id,       upper_arm_r.part_id, opt);
                Tji = calc_Tij(Lj, upper_arm_r.part_id, torso.part_id,       opt);
                dij = sum(abs(Tij - Tji));
                upper_arm_r_neighbors(1) = dij + upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind);
                
                Tij = calc_Tij(Li, torso.part_id,       upper_arm_l.part_id, opt);
                Tji = calc_Tij(Lj, upper_arm_l.part_id, torso.part_id,       opt);
                dij = sum(abs(Tij - Tji)); 
                upper_arm_l_neighbors(1) = dij + upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind);
                
                % neighbor in x dimension
                if (x_ind+1 <= opt.scan_nsample.x)
                    Lj = Li; Lj(1) = x_grid(x_ind+1);
                    
                    % for head part
                    Tij = calc_Tij(Li, torso.part_id, head.part_id,  opt);
                    Tji = calc_Tij(Lj, head.part_id,  torso.part_id, opt);
                    dij = sum(abs(Tij - Tji));
                    head_neighbors(2) = dij + head.B(x_ind+1, y_ind, theta_ind, s_ind) + opt.k.x;
                    
                    Tij = calc_Tij(Li, torso.part_id,       upper_arm_r.part_id, opt);
                    Tji = calc_Tij(Lj, upper_arm_r.part_id, torso.part_id,       opt);
                    dij = sum(abs(Tij - Tji));
                    upper_arm_r_neighbors(2) = dij + upper_arm_r.B(x_ind+1, y_ind, theta_ind, s_ind) + opt.k.x;
                    
                    Tij = calc_Tij(Li, torso.part_id,       upper_arm_l.part_id, opt);
                    Tji = calc_Tij(Lj, upper_arm_l.part_id, torso.part_id,       opt);
                    dij = sum(abs(Tij - Tji));                   
                    upper_arm_l_neighbors(2) = dij + upper_arm_l.B(x_ind+1, y_ind, theta_ind, s_ind) + opt.k.x;
                    
                    head_neighbors_lj(2, :) = head.Bj_p{x_ind+1, y_ind, theta_ind, s_ind};
                    upper_arm_r_neighbors_lj(2, :) = upper_arm_r.Bj_p{x_ind+1, y_ind, theta_ind, s_ind};
                    upper_arm_l_neighbors_lj(2, :) = upper_arm_l.Bj_p{x_ind+1, y_ind, theta_ind, s_ind};
                end
                
                % neighbor in y dimension
                if (y_ind+1 <= opt.scan_nsample.y) 
                    Lj = Li; Lj(2) = y_grid(y_ind+1);
                    
                    % for head part
                    Tij = calc_Tij(Li, torso.part_id, head.part_id,  opt);
                    Tji = calc_Tij(Lj, head.part_id,  torso.part_id, opt);
                    dij = sum(abs(Tij - Tji));
                    head_neighbors(3) = dij + head.B(x_ind, y_ind+1, theta_ind, s_ind) + opt.k.y;
                    
                    Tij = calc_Tij(Li, torso.part_id,       upper_arm_r.part_id, opt);
                    Tji = calc_Tij(Lj, upper_arm_r.part_id, torso.part_id,       opt);
                    dij = sum(abs(Tij - Tji));
                    upper_arm_r_neighbors(3) = dij + upper_arm_r.B(x_ind, y_ind+1, theta_ind, s_ind) + opt.k.y;
                    
                    Tij = calc_Tij(Li, torso.part_id,       upper_arm_l.part_id, opt);
                    Tji = calc_Tij(Lj, upper_arm_l.part_id, torso.part_id,       opt);
                    dij = sum(abs(Tij - Tji));                   
                    upper_arm_l_neighbors(3) = dij + upper_arm_l.B(x_ind, y_ind+1, theta_ind, s_ind) + opt.k.y;
                    
                    head_neighbors_lj(3, :) = head.Bj_p{x_ind, y_ind+1, theta_ind, s_ind};
                    upper_arm_r_neighbors_lj(3, :) = upper_arm_r.Bj_p{x_ind, y_ind+1, theta_ind, s_ind};
                    upper_arm_l_neighbors_lj(3, :) = upper_arm_l.Bj_p{x_ind, y_ind+1, theta_ind, s_ind};
                end
                
                % neighbor in theta dimension
                if (theta_ind+1 <= opt.scan_nsample.theta)
                    Lj = Li; Lj(3) = theta_grid(theta_ind+1);
                    
                    % for head part
                    Tij = calc_Tij(Li, torso.part_id, head.part_id,  opt);
                    Tji = calc_Tij(Lj, head.part_id,  torso.part_id, opt);
                    dij = sum(abs(Tij - Tji));
                    head_neighbors(4) = dij + head.B(x_ind, y_ind, theta_ind+1, s_ind) + opt.k.theta;
                    
                    Tij = calc_Tij(Li, torso.part_id,       upper_arm_r.part_id, opt);
                    Tji = calc_Tij(Lj, upper_arm_r.part_id, torso.part_id,       opt);
                    dij = sum(abs(Tij - Tji));
                    upper_arm_r_neighbors(4) = dij + upper_arm_r.B(x_ind, y_ind, theta_ind+1, s_ind) + opt.k.theta;
                    
                    Tij = calc_Tij(Li, torso.part_id,       upper_arm_l.part_id, opt);
                    Tji = calc_Tij(Lj, upper_arm_l.part_id, torso.part_id,       opt);
                    dij = sum(abs(Tij - Tji));                   
                    upper_arm_l_neighbors(4) = dij + upper_arm_l.B(x_ind, y_ind, theta_ind+1, s_ind) + opt.k.theta;
                    
                    head_neighbors_lj(4, :) = head.Bj_p{x_ind, y_ind, theta_ind+1, s_ind};
                    upper_arm_r_neighbors_lj(4, :) = upper_arm_r.Bj_p{x_ind, y_ind, theta_ind+1, s_ind};
                    upper_arm_l_neighbors_lj(4, :) = upper_arm_l.Bj_p{x_ind, y_ind, theta_ind+1, s_ind};
                end
                
                % neighbor in s dimension
                if (s_ind+1 <= opt.scan_nsample.s) 
                    Lj = Li; Lj(4) = s_grid(s_ind+1);
                    
                    % for head part
                    Tij = calc_Tij(Li, torso.part_id, head.part_id,  opt);
                    Tji = calc_Tij(Lj, head.part_id,  torso.part_id, opt);
                    dij = sum(abs(Tij - Tji));
                    head_neighbors(5) = dij + head.B(x_ind, y_ind, theta_ind, s_ind+1) + opt.k.s;
                    
                    Tij = calc_Tij(Li, torso.part_id,       upper_arm_r.part_id, opt);
                    Tji = calc_Tij(Lj, upper_arm_r.part_id, torso.part_id,       opt);
                    dij = sum(abs(Tij - Tji));
                    upper_arm_r_neighbors(5) = dij + upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind+1) + opt.k.s;
                    
                    Tij = calc_Tij(Li, torso.part_id,       upper_arm_l.part_id, opt);
                    Tji = calc_Tij(Lj, upper_arm_l.part_id, torso.part_id,       opt);
                    dij = sum(abs(Tij - Tji));                   
                    upper_arm_l_neighbors(5) = dij + upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind+1) + opt.k.s;
                    
                    head_neighbors_lj(5, :) = head.Bj_p{x_ind, y_ind, theta_ind, s_ind+1};
                    upper_arm_r_neighbors_lj(5, :) = upper_arm_r.Bj_p{x_ind, y_ind, theta_ind, s_ind+1};
                    upper_arm_l_neighbors_lj(5, :) = upper_arm_l.Bj_p{x_ind, y_ind, theta_ind, s_ind+1};
                end
                
                % obtain the min/argmin value
                [head.B(x_ind, y_ind, theta_ind, s_ind), head_min_ind]           = min(head_neighbors);
                [upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind), upper_arm_r_min_ind] = min(upper_arm_r_neighbors);
                [upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind), upper_arm_l_min_ind] = min(upper_arm_l_neighbors);
                
                head.Bj_p{x_ind, y_ind, theta_ind, s_ind} = head_neighbors_lj(head_min_ind, :);
                upper_arm_r.Bj_p{x_ind, y_ind, theta_ind, s_ind} = upper_arm_r_neighbors_lj(upper_arm_r_min_ind, :);
                upper_arm_l.Bj_p{x_ind, y_ind, theta_ind, s_ind} = upper_arm_l_neighbors_lj(upper_arm_l_min_ind, :);
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Backward pass through D to find the minimum value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nBackward pass through D for all leave nodes...\n');
for x_ind = length(x_grid) : -1 : 1
    
    % display progress
    if (mod(x_ind,5) == 0) 
        fprintf('Progress: %.0f%%\n', 100*(length(x_grid)-x_ind+1)/length(x_grid)); 
    end
    
    for y_ind = length(y_grid) : -1 : 1
        for theta_ind = length(theta_grid) : -1 : 1
            for s_ind = length(s_grid) : -1 : 1
                
                                % initialize neighbor vectors
                head_neighbors = Inf(1, 5); % for min value
                upper_arm_r_neighbors = Inf(1, 5);
                upper_arm_l_neighbors = Inf(1, 5);
                
                head_neighbors_lj = zeros(5, 4); % for argmin value, 5-neighbors
                upper_arm_r_neighbors_lj = zeros(5, 4);
                upper_arm_l_neighbors_lj = zeros(5, 4);
                
                % retrieve the coordinate
                x = x_grid(x_ind);
                y = y_grid(y_ind);
                s = s_grid(s_ind);
                theta = theta_grid(theta_ind);
                
                Li = [x, y, theta, s]; % Li
                
                % retrieve neighboring elements
                
                % include itself as well
                Lj = Li;
                Tij = calc_Tij(Li, torso.part_id, head.part_id,  opt);
                Tji = calc_Tij(Lj, head.part_id,  torso.part_id, opt);
                dij = sum(abs(Tij - Tji));
                head_neighbors(1) = dij + head.B(x_ind, y_ind, theta_ind, s_ind);
                
                Tij = calc_Tij(Li, torso.part_id,       upper_arm_r.part_id, opt);
                Tji = calc_Tij(Lj, upper_arm_r.part_id, torso.part_id,       opt);
                dij = sum(abs(Tij - Tji));
                upper_arm_r_neighbors(1) = dij + upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind);
                
                Tij = calc_Tij(Li, torso.part_id,       upper_arm_l.part_id, opt);
                Tji = calc_Tij(Lj, upper_arm_l.part_id, torso.part_id,       opt);
                dij = sum(abs(Tij - Tji)); 
                upper_arm_l_neighbors(1) = dij + upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind);
                
                % neighbor in x dimension
                if (x_ind-1 >= 1)
                    Lj = Li; Lj(1) = x_grid(x_ind-1);
                    
                    % for head part
                    Tij = calc_Tij(Li, torso.part_id, head.part_id,  opt);
                    Tji = calc_Tij(Lj, head.part_id,  torso.part_id, opt);
                    dij = sum(abs(Tij - Tji));
                    head_neighbors(2) = dij + head.B(x_ind-1, y_ind, theta_ind, s_ind) + opt.k.x;
                    
                    Tij = calc_Tij(Li, torso.part_id,       upper_arm_r.part_id, opt);
                    Tji = calc_Tij(Lj, upper_arm_r.part_id, torso.part_id,       opt);
                    dij = sum(abs(Tij - Tji));
                    upper_arm_r_neighbors(2) = dij + upper_arm_r.B(x_ind-1, y_ind, theta_ind, s_ind) + opt.k.x;
                    
                    Tij = calc_Tij(Li, torso.part_id,       upper_arm_l.part_id, opt);
                    Tji = calc_Tij(Lj, upper_arm_l.part_id, torso.part_id,       opt);
                    dij = sum(abs(Tij - Tji));                   
                    upper_arm_l_neighbors(2) = dij + upper_arm_l.B(x_ind-1, y_ind, theta_ind, s_ind) + opt.k.x;
                    
                    head_neighbors_lj(2, :) = head.Bj_p{x_ind-1, y_ind, theta_ind, s_ind};
                    upper_arm_r_neighbors_lj(2, :) = upper_arm_r.Bj_p{x_ind-1, y_ind, theta_ind, s_ind};
                    upper_arm_l_neighbors_lj(2, :) = upper_arm_l.Bj_p{x_ind-1, y_ind, theta_ind, s_ind};
                end
                
                % neighbor in y dimension
                if (y_ind-1 >= 1) 
                    Lj = Li; Lj(2) = y_grid(y_ind-1);
                    
                    % for head part
                    Tij = calc_Tij(Li, torso.part_id, head.part_id,  opt);
                    Tji = calc_Tij(Lj, head.part_id,  torso.part_id, opt);
                    dij = sum(abs(Tij - Tji));
                    head_neighbors(3) = dij + head.B(x_ind, y_ind-1, theta_ind, s_ind) + opt.k.y;
                    
                    Tij = calc_Tij(Li, torso.part_id,       upper_arm_r.part_id, opt);
                    Tji = calc_Tij(Lj, upper_arm_r.part_id, torso.part_id,       opt);
                    dij = sum(abs(Tij - Tji));
                    upper_arm_r_neighbors(3) = dij + upper_arm_r.B(x_ind, y_ind-1, theta_ind, s_ind) + opt.k.y;
                    
                    Tij = calc_Tij(Li, torso.part_id,       upper_arm_l.part_id, opt);
                    Tji = calc_Tij(Lj, upper_arm_l.part_id, torso.part_id,       opt);
                    dij = sum(abs(Tij - Tji));                   
                    upper_arm_l_neighbors(3) = dij + upper_arm_l.B(x_ind, y_ind-1, theta_ind, s_ind) + opt.k.y;
                    
                    head_neighbors_lj(3, :) = head.Bj_p{x_ind, y_ind-1, theta_ind, s_ind};
                    upper_arm_r_neighbors_lj(3, :) = upper_arm_r.Bj_p{x_ind, y_ind-1, theta_ind, s_ind};
                    upper_arm_l_neighbors_lj(3, :) = upper_arm_l.Bj_p{x_ind, y_ind-1, theta_ind, s_ind};
                end
                
                % neighbor in theta dimension
                if (theta_ind-1 >= 1)
                    Lj = Li; Lj(3) = theta_grid(theta_ind-1);
                    
                    % for head part
                    Tij = calc_Tij(Li, torso.part_id, head.part_id,  opt);
                    Tji = calc_Tij(Lj, head.part_id,  torso.part_id, opt);
                    dij = sum(abs(Tij - Tji));
                    head_neighbors(4) = dij + head.B(x_ind, y_ind, theta_ind-1, s_ind) + opt.k.theta;
                    
                    Tij = calc_Tij(Li, torso.part_id,       upper_arm_r.part_id, opt);
                    Tji = calc_Tij(Lj, upper_arm_r.part_id, torso.part_id,       opt);
                    dij = sum(abs(Tij - Tji));
                    upper_arm_r_neighbors(4) = dij + upper_arm_r.B(x_ind, y_ind, theta_ind-1, s_ind) + opt.k.theta;
                    
                    Tij = calc_Tij(Li, torso.part_id,       upper_arm_l.part_id, opt);
                    Tji = calc_Tij(Lj, upper_arm_l.part_id, torso.part_id,       opt);
                    dij = sum(abs(Tij - Tji));                   
                    upper_arm_l_neighbors(4) = dij + upper_arm_l.B(x_ind, y_ind, theta_ind-1, s_ind) + opt.k.theta;
                    
                    head_neighbors_lj(4, :) = head.Bj_p{x_ind, y_ind, theta_ind-1, s_ind};
                    upper_arm_r_neighbors_lj(4, :) = upper_arm_r.Bj_p{x_ind, y_ind, theta_ind-1, s_ind};
                    upper_arm_l_neighbors_lj(4, :) = upper_arm_l.Bj_p{x_ind, y_ind, theta_ind-1, s_ind};
                end
                
                % neighbor in s dimension
                if (s_ind-1 >= 1) 
                    Lj = Li; Lj(4) = s_grid(s_ind-1);
                    
                    % for head part
                    Tij = calc_Tij(Li, torso.part_id, head.part_id,  opt);
                    Tji = calc_Tij(Lj, head.part_id,  torso.part_id, opt);
                    dij = sum(abs(Tij - Tji));
                    head_neighbors(5) = dij + head.B(x_ind, y_ind, theta_ind, s_ind-1) + opt.k.s;
                    
                    Tij = calc_Tij(Li, torso.part_id,       upper_arm_r.part_id, opt);
                    Tji = calc_Tij(Lj, upper_arm_r.part_id, torso.part_id,       opt);
                    dij = sum(abs(Tij - Tji));
                    upper_arm_r_neighbors(5) = dij + upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind-1) + opt.k.s;
                    
                    Tij = calc_Tij(Li, torso.part_id,       upper_arm_l.part_id, opt);
                    Tji = calc_Tij(Lj, upper_arm_l.part_id, torso.part_id,       opt);
                    dij = sum(abs(Tij - Tji));                   
                    upper_arm_l_neighbors(5) = dij + upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind-1) + opt.k.s;
                    
                    head_neighbors_lj(5, :) = head.Bj_p{x_ind, y_ind, theta_ind, s_ind-1};
                    upper_arm_r_neighbors_lj(5, :) = upper_arm_r.Bj_p{x_ind, y_ind, theta_ind, s_ind-1};
                    upper_arm_l_neighbors_lj(5, :) = upper_arm_l.Bj_p{x_ind, y_ind, theta_ind, s_ind-1};
                end
                
                % obtain the min/argmin value
                [head.B(x_ind, y_ind, theta_ind, s_ind), head_min_ind]           = min(head_neighbors);
                [upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind), upper_arm_r_min_ind] = min(upper_arm_r_neighbors);
                [upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind), upper_arm_l_min_ind] = min(upper_arm_l_neighbors);
                
                head.Bj_p{x_ind, y_ind, theta_ind, s_ind} = head_neighbors_lj(head_min_ind, :);
                upper_arm_r.Bj_p{x_ind, y_ind, theta_ind, s_ind} = upper_arm_r_neighbors_lj(upper_arm_r_min_ind, :);
                upper_arm_l.Bj_p{x_ind, y_ind, theta_ind, s_ind} = upper_arm_l_neighbors_lj(upper_arm_l_min_ind, :);
                
            end
        end
    end
end

%% Compute f(w) for distance transformation and initialize D for root node

fprintf('Initializing D using f(w) for all leave nodes...\n');

torso_opt_L = [1, 1, 1, 1];
torso_opt_E = Inf;
for x_ind = 1 : length(x_grid)
    
    % display progress
    fprintf('Progress: %.0f%%\n', 100*x_ind/length(x_grid)); 
    
    for y_ind = 1 : length(y_grid)
        for theta_ind = 1 : length(theta_grid)
            for s_ind = 1 : length(s_grid)
                
                % retrieve the coordinate and energy of leave nodes
                x = x_grid(x_ind);
                y = y_grid(y_ind);
                s = s_grid(s_ind);
                theta = theta_grid(theta_ind);
                
                L = [x, y, theta, s];
                head_energy = head.B(x_ind, y_ind, theta_ind, s_ind);
                upper_arm_r_energy = upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind);
                upper_arm_l_energy = upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind);
                
                % compute the total energy
                torso.B(x_ind, y_ind, theta_ind, s_ind) = ...
                    match_energy_cost(L, torso.part_id, dat_pt(:,torso.part_id)) + ...
                    head_energy + upper_arm_r_energy + upper_arm_r_energy; 
                
                % update the minimum value
                if (torso.B(x_ind, y_ind, theta_ind, s_ind) < torso_opt_E)
                    torso_opt_E = torso.B(x_ind, y_ind, theta_ind, s_ind);
                    torso_opt_L = [x_ind, y_ind, theta_ind, s_ind];
                end
            end
        end
    end
end

%% Obtain the coordinate of each node
% torso
torso_x = x_grid(torso_opt_L(1));
torso_y = y_grid(torso_opt_L(2));
torso_theta = theta_grid(torso_opt_L(3));
torso_s = s_grid(torso_opt_L(4));
torso_corr = [torso_x - torso_s*opt.model.len(1)/2*cos(torso_theta); ...
    torso_x + torso_s*opt.model.len(1)/2*cos(torso_theta); ...
    torso_y - torso_s*opt.model.len(1)/2*sin(torso_theta); ...
    torso_y + torso_s*opt.model.len(1)/2*sin(torso_theta)];

% left-upper arm
upper_arm_l_corr = upper_arm_l.Bj_p{torso_opt_L(1), torso_opt_L(2), torso_opt_L(3), torso_opt_L(4)};
upper_arm_l_corr = [upper_arm_l_corr(1) - upper_arm_l_corr(4)*opt.model.len(2)/2*cos(upper_arm_l_corr(3)); ...
    upper_arm_l_corr(1) + upper_arm_l_corr(4)*opt.model.len(2)/2*cos(upper_arm_l_corr(3)); ...
    upper_arm_l_corr(2) - upper_arm_l_corr(4)*opt.model.len(2)/2*sin(upper_arm_l_corr(3)); ...
    upper_arm_l_corr(2) + upper_arm_l_corr(4)*opt.model.len(2)/2*sin(upper_arm_l_corr(3))];

% right-upper arm
upper_arm_r_corr = upper_arm_r.Bj_p{torso_opt_L(1), torso_opt_L(2), torso_opt_L(3), torso_opt_L(4)};
upper_arm_r_corr = [upper_arm_r_corr(1) - upper_arm_r_corr(4)*opt.model.len(3)/2*cos(upper_arm_r_corr(3)); ...
    upper_arm_r_corr(1) + upper_arm_r_corr(4)*opt.model.len(3)/2*cos(upper_arm_r_corr(3)); ...
    upper_arm_r_corr(2) - upper_arm_r_corr(4)*opt.model.len(3)/2*sin(upper_arm_r_corr(3)); ...
    upper_arm_r_corr(2) + upper_arm_r_corr(4)*opt.model.len(3)/2*sin(upper_arm_r_corr(3))];

% head
head_corr = head.Bj_p{torso_opt_L(1), torso_opt_L(2), torso_opt_L(3), torso_opt_L(4)};
head_corr = [head_corr(1) - head_corr(4)*opt.model.len(6)/2*cos(head_corr(3)); ...
    head_corr(1) + head_corr(4)*opt.model.len(6)/2*cos(head_corr(3)); ...
    head_corr(2) - head_corr(4)*opt.model.len(6)/2*sin(head_corr(3)); ...
    head_corr(2) + head_corr(4)*opt.model.len(6)/2*sin(head_corr(3))];

% 4 part corr
total_corr = [torso_corr upper_arm_l_corr upper_arm_r_corr head_corr];

% Draw stickman
colors = [0.99 0 0 0; 0 0.99 0.99 0; 0 0 0 0.99]; 
% torso - red; arms - green; head - blue
thickness = 4;
drawidx = true;
drawfullskeleton = 1;
hdl = DrawStickman(total_corr, img, colors, thickness, drawidx);



% %% Compute minimum distance in D for root node
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Forward pass through D to find the minimum value
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% fprintf('\nForward pass through D for the root node...\n');
% 
% % keep track of the lowest energy and the associated configuration
% torso_opt_L = [1, 1, 1, 1];
% torso_opt_E = Inf;
% for x_ind = 1 : length(x_grid)
%     
%     % display progress
%     if (mod(x_ind,5) == 0) 
%         fprintf('Progress: %.0f%%\n', 100*x_ind/length(x_grid)); 
%     end
%     
%     for y_ind = 1 : length(y_grid)
%         for theta_ind = 1 : length(theta_grid)
%             for s_ind = 1 : length(s_grid)
%                 
%                 % initialize neighbor vectors
%                 neighbors = Inf(1, 5);
%                 
%                 % retrieve neighboring elements
%                 neighbors(1) = head.B(x_ind, y_ind, theta_ind, s_ind);
%                 
%                 if (x_ind+1 <= opt.scan_nsample.x) % neighbor in x dimension
%                     neighbors(2) = torso.B(x_ind+1, y_ind, theta_ind, s_ind) + opt.k.x;
%                 end
%                 if (y_ind+1 <= opt.scan_nsample.y) % neighbor in y dimension
%                     neighbors(3) = torso.B(x_ind, y_ind+1, theta_ind, s_ind) + opt.k.y;
%                 end
%                 if (theta_ind+1 <= opt.scan_nsample.theta) % neighbor in theta dimension
%                     neighbors(4) = torso.B(x_ind, y_ind, theta_ind+1, s_ind) + opt.k.theta;
%                 end
%                 if (s_ind+1 <= opt.scan_nsample.s) % neighbor in s dimension
%                     neighbors(5) = torso.B(x_ind, y_ind, theta_ind, s_ind+1) + opt.k.s;
%                 end
%                 
%                 % obtain the min value
%                 torso.B(x_ind, y_ind, theta_ind, s_ind) = min(neighbors);
%                 
%                 % update the min value and coordinate
%                 if (torso.B(x_ind, y_ind, theta_ind, s_ind) < torso_opt_E)
%                     torso_opt_L = [x_ind, y_ind, theta_ind, s_ind];
%                     torso_opt_E = torso.B(x_ind, y_ind, theta_ind, s_ind);
%                 end
%             end
%         end
%     end
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Backward pass through D to find the minimum value
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % do not initialize minimum value and coordinate at this time
% fprintf('\nBackward pass through D for all leave nodes...\n');
% for x_ind = length(x_grid) : -1 : 1
%     
%     % display progress
%     if (mod(x_ind,5) == 0) 
%         fprintf('Progress: %.0f%%\n', 100*(length(x_grid)-x_ind+1)/length(x_grid)); 
%     end
%     
%     for y_ind = length(y_grid) : -1 : 1
%         for theta_ind = length(theta_grid) : -1 : 1
%             for s_ind = length(s_grid) : -1 : 1
%                 
%                 % initialize neighbor vectors
%                 neighbors = Inf(1, 5);
%                 
%                 % retrieve neighboring elements
%                 neighbors(1) = head.B(x_ind, y_ind, theta_ind, s_ind);
%                 
%                 if (x_ind-1 >= 1) % neighbor in x dimension
%                     neighbors(2) = torso.B(x_ind-1, y_ind, theta_ind, s_ind) + opt.k.x;
%                 end
%                 if (y_ind-1 >= 1) % neighbor in y dimension
%                     neighbors(3) = torso.B(x_ind, y_ind-1, theta_ind, s_ind) + opt.k.y;
%                 end
%                 if (theta_ind-1 >= 1) % neighbor in theta dimension
%                     neighbors(4) = torso.B(x_ind, y_ind, theta_ind-1, s_ind) + opt.k.theta;
%                 end
%                 if (s_ind-1 >= 1) % neighbor in s dimension
%                     neighbors(5) = torso.B(x_ind, y_ind, theta_ind, s_ind-1) + opt.k.s;
%                 end
%                 
%                 % obtain the min value
%                 torso.B(x_ind, y_ind, theta_ind, s_ind) = min(neighbors);
%                 
%                 % update the min value and coordinate
%                 if (torso.B(x_ind, y_ind, theta_ind, s_ind) < torso_opt_E)
%                     torso_opt_L = [x_ind, y_ind, theta_ind, s_ind];
%                     torso_opt_E = torso.B(x_ind, y_ind, theta_ind, s_ind);
%                 end
%                 
%             end
%         end
%     end
% end


