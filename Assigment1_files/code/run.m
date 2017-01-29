clc; close all; clear all;

%% Initialization

fprintf('Initializing...\n');

% Startup command
installmex;
startup;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% specify and read in the image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
img_seq = 5;   % the sequence (index) of the image in the folder
img_id  = 829; % the id of the image filename
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

% Create parameter struct opt
set_opt;

% Specify the search grid in each dimension
[m, n, ~] = size(img);
x_grid = linspace(1, n, opt.scan_nsample.x);
y_grid = linspace(1, m, opt.scan_nsample.y);
theta_grid = linspace(opt.scan_nsample.theta_min, opt.scan_nsample.theta_max, opt.scan_nsample.theta); % normalization???
s_grid = linspace(opt.scan_nsample.s_min, opt.scan_nsample.s_max, opt.scan_nsample.s);

% Initialize entire searching space
search_grid_dim = [opt.scan_nsample.x, opt.scan_nsample.y, opt.scan_nsample.theta, opt.scan_nsample.s];
search_grid = combvec(x_grid, y_grid, theta_grid, s_grid);
n_search = size(search_grid, 2);

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pre-compute Tij and Tji for acceleration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T_head_torso = calc_Tij(head.part_id, torso.part_id, x_grid, y_grid, theta_grid, s_grid, opt);
T_upper_arm_l_torso = calc_Tij(upper_arm_l.part_id, torso.part_id, x_grid, y_grid, theta_grid, s_grid, opt);
T_upper_arm_r_torso = calc_Tij(upper_arm_r.part_id, torso.part_id, x_grid, y_grid, theta_grid, s_grid, opt);

T_torso_head = calc_Tij(torso.part_id, head.part_id, x_grid, y_grid, theta_grid, s_grid, opt);
T_torso_upper_arm_l = calc_Tij(torso.part_id, upper_arm_l.part_id, x_grid, y_grid, theta_grid, s_grid, opt);
T_torso_upper_arm_r = calc_Tij(torso.part_id, upper_arm_r.part_id, x_grid, y_grid, theta_grid, s_grid, opt);

%% Compute f(w) for distance transformation and initialize D for leave nodes

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initializing D using f(w) for all leave node
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Initializing D using f(w) for all leave nodes...\n');

for l_ind = 1 : n_search
    if (mod(l_ind, 50000) == 0)
        fprintf('f(w) Initialization Progress: %.0f%%\n', 100*l_ind/size(search_grid, 2)); 
    end
    
    Lj = search_grid(:, l_ind)';
    [x_ind, y_ind, theta_ind, s_ind] = ind2sub(search_grid_dim, l_ind);

    head.B(x_ind, y_ind, theta_ind, s_ind) = match_energy_cost(Lj, head.part_id, dat_pt(:,head.part_id));
    upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind) = match_energy_cost(Lj, upper_arm_l.part_id, dat_pt(:,upper_arm_l.part_id)); 
    upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind) = match_energy_cost(Lj, upper_arm_r.part_id, dat_pt(:,upper_arm_r.part_id));

    head.Bj_p{x_ind, y_ind, theta_ind, s_ind} = Lj;
    upper_arm_l.Bj_p{x_ind, y_ind, theta_ind, s_ind} = Lj;
    upper_arm_r.Bj_p{x_ind, y_ind, theta_ind, s_ind} = Lj;
end

% After acceleration + parfor: 17.416040 seconds (with the cost of losing readability)
% After acceleration:          27.939928 seconds
% Before acceleration:         51.130546 seconds

%% Testcase for f(w) initialization
% B = upper_arm_l.B;
% [val, ind] = min(B(:));
% [x_ind, y_ind, theta_ind, s_ind] = ind2sub(search_grid_dim, ind);
% fprintf('Optimal x: %d\n', x_grid(x_ind));
% fprintf('Optimal y: %d\n', y_grid(y_ind));
% fprintf('Optimal theta: %d\n', theta_grid(theta_ind));
% fprintf('Optimal s: %d\n', s_grid(s_ind));

%% Compute minimum distance in D for leave nodes (Forward Pass)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Forward pass through D to find the minimum value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nForward pass through D for all leave nodes...\n');
for l_ind = 1 : n_search % seach for all possibility of li

    if (mod(l_ind, 5000) == 0)
        fprintf('Leave Node Forward Pass Progress: %.0f%%\n', 100*l_ind/size(search_grid, 2)); 
    end

    [x_ind, y_ind, theta_ind, s_ind] = ind2sub(search_grid_dim, l_ind);
    
    % initialize neighbor vectors
    head_neighbors = Inf(1, 5); % for min value
    upper_arm_r_neighbors = Inf(1, 5);
    upper_arm_l_neighbors = Inf(1, 5);

    head_neighbors_lj = zeros(5, 4); % for argmin value, 5-neighbors
    upper_arm_r_neighbors_lj = zeros(5, 4);
    upper_arm_l_neighbors_lj = zeros(5, 4);
    
    %%%%%%%%%%%
    % Lj = Li
    %%%%%%%%%%%
    Li = l_ind; 
    Lj = l_ind;
    
    Tij = T_head_torso(:, Li);
    Tji = T_torso_head(:, Lj);
    dij = sum(abs(Tij - Tji));
    head_neighbors(1) = dij + head.B(x_ind, y_ind, theta_ind, s_ind);

    Tij = T_upper_arm_l_torso(:, Li);
    Tji = T_torso_upper_arm_l(:, Lj);
    dij = sum(abs(Tij - Tji));
    upper_arm_l_neighbors(1) = dij + upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind);

    Tij = T_upper_arm_r_torso(:, Li);
    Tji = T_torso_upper_arm_r(:, Lj);
    dij = sum(abs(Tij - Tji));
    upper_arm_r_neighbors(1) = dij + upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind);
    
    % take down the lj that gives this distance
    head_neighbors_lj(1, :) = head.Bj_p{x_ind, y_ind, theta_ind, s_ind};
    upper_arm_r_neighbors_lj(1, :) = upper_arm_r.Bj_p{x_ind, y_ind, theta_ind, s_ind};
    upper_arm_l_neighbors_lj(1, :) = upper_arm_l.Bj_p{x_ind, y_ind, theta_ind, s_ind};

    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % neighbor in x-direction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (x_ind+1 <= opt.scan_nsample.x)
        
        Li = l_ind;
        Lj = sub2ind(search_grid_dim, x_ind+1, y_ind, theta_ind, s_ind);
        
        % compute corresponding distance
        Tij = T_head_torso(:, Li);
        Tji = T_torso_head(:, Lj);
        dij = sum(abs(Tij - Tji));
        head_neighbors(2) = dij + head.B(x_ind+1, y_ind, theta_ind, s_ind);
        
        Tij = T_upper_arm_r_torso(:, Li);
        Tji = T_torso_upper_arm_r(:, Lj);
        dij = sum(abs(Tij - Tji));
        upper_arm_r_neighbors(2) = dij + upper_arm_r.B(x_ind+1, y_ind, theta_ind, s_ind);

        Tij = T_upper_arm_l_torso(:, Li);
        Tji = T_torso_upper_arm_l(:, Lj);
        dij = sum(abs(Tij - Tji));
        upper_arm_l_neighbors(2) = dij + upper_arm_l.B(x_ind+1, y_ind, theta_ind, s_ind);
        
        % take down the lj that gives this distance
        head_neighbors_lj(2, :) = head.Bj_p{x_ind+1, y_ind, theta_ind, s_ind};
        upper_arm_r_neighbors_lj(2, :) = upper_arm_r.Bj_p{x_ind+1, y_ind, theta_ind, s_ind};
        upper_arm_l_neighbors_lj(2, :) = upper_arm_l.Bj_p{x_ind+1, y_ind, theta_ind, s_ind};
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % neighbor in y-direction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (y_ind+1 <= opt.scan_nsample.y) 
        
        Li = l_ind;
        Lj = sub2ind(search_grid_dim, x_ind, y_ind+1, theta_ind, s_ind);
        
        % compute corresponding distance
        Tij = T_head_torso(:, Li);
        Tji = T_torso_head(:, Lj);
        dij = sum(abs(Tij - Tji));
        head_neighbors(3) = dij + head.B(x_ind, y_ind+1, theta_ind, s_ind);
        
        Tij = T_upper_arm_r_torso(:, Li);
        Tji = T_torso_upper_arm_r(:, Lj);
        dij = sum(abs(Tij - Tji));
        upper_arm_r_neighbors(3) = dij + upper_arm_r.B(x_ind, y_ind+1, theta_ind, s_ind);

        Tij = T_upper_arm_l_torso(:, Li);
        Tji = T_torso_upper_arm_l(:, Lj);
        dij = sum(abs(Tij - Tji));
        upper_arm_l_neighbors(3) = dij + upper_arm_l.B(x_ind, y_ind+1, theta_ind, s_ind);
        
        % take down the lj that gives this distance
        head_neighbors_lj(3, :) = head.Bj_p{x_ind, y_ind+1, theta_ind, s_ind};
        upper_arm_r_neighbors_lj(3, :) = upper_arm_r.Bj_p{x_ind, y_ind+1, theta_ind, s_ind};
        upper_arm_l_neighbors_lj(3, :) = upper_arm_l.Bj_p{x_ind, y_ind+1, theta_ind, s_ind};
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % neighbor in theta-direction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (theta_ind+1 <= opt.scan_nsample.theta)
        
        Li = l_ind;
        Lj = sub2ind(search_grid_dim, x_ind, y_ind, theta_ind+1, s_ind);
        
        % compute corresponding distance
        Tij = T_head_torso(:, Li);
        Tji = T_torso_head(:, Lj);
        dij = sum(abs(Tij - Tji));
        head_neighbors(4) = dij + head.B(x_ind, y_ind, theta_ind+1, s_ind);
        
        Tij = T_upper_arm_r_torso(:, Li);
        Tji = T_torso_upper_arm_r(:, Lj);
        dij = sum(abs(Tij - Tji));
        upper_arm_r_neighbors(4) = dij + upper_arm_r.B(x_ind, y_ind, theta_ind+1, s_ind);

        Tij = T_upper_arm_l_torso(:, Li);
        Tji = T_torso_upper_arm_l(:, Lj);
        dij = sum(abs(Tij - Tji));
        upper_arm_l_neighbors(4) = dij + upper_arm_l.B(x_ind, y_ind, theta_ind+1, s_ind);
        
        % take down the lj that gives this distance
        head_neighbors_lj(4, :) = head.Bj_p{x_ind, y_ind, theta_ind+1, s_ind};
        upper_arm_r_neighbors_lj(4, :) = upper_arm_r.Bj_p{x_ind, y_ind, theta_ind+1, s_ind};
        upper_arm_l_neighbors_lj(4, :) = upper_arm_l.Bj_p{x_ind, y_ind, theta_ind+1, s_ind};
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % neighbor in s-direction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (s_ind+1 <= opt.scan_nsample.s) 
        
        Li = l_ind;
        Lj = sub2ind(search_grid_dim, x_ind, y_ind, theta_ind, s_ind+1);
        
        % compute corresponding distance
        Tij = T_head_torso(:, Li);
        Tji = T_torso_head(:, Lj);
        dij = sum(abs(Tij - Tji));
        head_neighbors(5) = dij + head.B(x_ind, y_ind, theta_ind, s_ind+1);
        
        Tij = T_upper_arm_r_torso(:, Li);
        Tji = T_torso_upper_arm_r(:, Lj);
        dij = sum(abs(Tij - Tji));
        upper_arm_r_neighbors(5) = dij + upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind+1);

        Tij = T_upper_arm_l_torso(:, Li);
        Tji = T_torso_upper_arm_l(:, Lj);
        dij = sum(abs(Tij - Tji));
        upper_arm_l_neighbors(5) = dij + upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind+1);
        
        % take down the lj that gives this distance
        head_neighbors_lj(5, :) = head.Bj_p{x_ind, y_ind, theta_ind, s_ind+1};
        upper_arm_r_neighbors_lj(5, :) = upper_arm_r.Bj_p{x_ind, y_ind, theta_ind, s_ind+1};
        upper_arm_l_neighbors_lj(5, :) = upper_arm_l.Bj_p{x_ind, y_ind, theta_ind, s_ind+1};
    end

    % obtain the min/argmin value
    [head.B(x_ind, y_ind, theta_ind, s_ind), head_min_ind] = min(head_neighbors);
    [upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind), upper_arm_r_min_ind] = min(upper_arm_r_neighbors);
    [upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind), upper_arm_l_min_ind] = min(upper_arm_l_neighbors);

    head.Bj_p{x_ind, y_ind, theta_ind, s_ind} = head_neighbors_lj(head_min_ind, :);
    upper_arm_r.Bj_p{x_ind, y_ind, theta_ind, s_ind} = upper_arm_r_neighbors_lj(upper_arm_r_min_ind, :);
    upper_arm_l.Bj_p{x_ind, y_ind, theta_ind, s_ind} = upper_arm_l_neighbors_lj(upper_arm_l_min_ind, :);

end

% Single Pass after pre-computation: 47.112885 seconds
% Single Pass after acceleration: 218.598945 seconds
% Single Pass before acceleration: ~8 min

%% Compute minimum distance in D for leave nodes (Backward Pass)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Backward pass through D to find the minimum value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nBackward pass through D for all leave nodes...\n');
for l_ind = n_search : -1 : 1    

    if (mod(l_ind, 5000) == 0)
        fprintf('Leave Node Backward Pass Progress: %.0f%%\n', 100*(n_search-l_ind+5000)/size(search_grid, 2)); 
    end

    [x_ind, y_ind, theta_ind, s_ind] = ind2sub(search_grid_dim, l_ind);
    
    % initialize neighbor vectors
    head_neighbors = Inf(1, 5); % for min value
    upper_arm_r_neighbors = Inf(1, 5);
    upper_arm_l_neighbors = Inf(1, 5);

    head_neighbors_lj = zeros(5, 4); % for argmin value, 5-neighbors
    upper_arm_r_neighbors_lj = zeros(5, 4);
    upper_arm_l_neighbors_lj = zeros(5, 4);
    
    %%%%%%%%%%%
    % Lj = Li
    %%%%%%%%%%%
    Li = l_ind; 
    Lj = l_ind;
    
    Tij = T_head_torso(:, Li);
    Tji = T_torso_head(:, Lj);
    dij = sum(abs(Tij - Tji));
    head_neighbors(1) = dij + head.B(x_ind, y_ind, theta_ind, s_ind);

    Tij = T_upper_arm_l_torso(:, Li);
    Tji = T_torso_upper_arm_l(:, Lj);
    dij = sum(abs(Tij - Tji));
    upper_arm_l_neighbors(1) = dij + upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind);

    Tij = T_upper_arm_r_torso(:, Li);
    Tji = T_torso_upper_arm_r(:, Lj);
    dij = sum(abs(Tij - Tji));
    upper_arm_r_neighbors(1) = dij + upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind);
    
    % take down the lj that gives this distance
    head_neighbors_lj(1, :) = head.Bj_p{x_ind, y_ind, theta_ind, s_ind};
    upper_arm_r_neighbors_lj(1, :) = upper_arm_r.Bj_p{x_ind, y_ind, theta_ind, s_ind};
    upper_arm_l_neighbors_lj(1, :) = upper_arm_l.Bj_p{x_ind, y_ind, theta_ind, s_ind};

    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % neighbor in x-direction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (x_ind-1 >= 1)
        
        Li = l_ind;
        Lj = sub2ind(search_grid_dim, x_ind-1, y_ind, theta_ind, s_ind);
        
        % compute corresponding distance
        Tij = T_head_torso(:, Li);
        Tji = T_torso_head(:, Lj);
        dij = sum(abs(Tij - Tji));
        head_neighbors(2) = dij + head.B(x_ind-1, y_ind, theta_ind, s_ind);
        
        Tij = T_upper_arm_r_torso(:, Li);
        Tji = T_torso_upper_arm_r(:, Lj);
        dij = sum(abs(Tij - Tji));
        upper_arm_r_neighbors(2) = dij + upper_arm_r.B(x_ind-1, y_ind, theta_ind, s_ind);

        Tij = T_upper_arm_l_torso(:, Li);
        Tji = T_torso_upper_arm_l(:, Lj);
        dij = sum(abs(Tij - Tji));
        upper_arm_l_neighbors(2) = dij + upper_arm_l.B(x_ind-1, y_ind, theta_ind, s_ind);
        
        % take down the lj that gives this distance
        head_neighbors_lj(2, :) = head.Bj_p{x_ind-1, y_ind, theta_ind, s_ind};
        upper_arm_r_neighbors_lj(2, :) = upper_arm_r.Bj_p{x_ind-1, y_ind, theta_ind, s_ind};
        upper_arm_l_neighbors_lj(2, :) = upper_arm_l.Bj_p{x_ind-1, y_ind, theta_ind, s_ind};
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % neighbor in y-direction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (y_ind-1 >= 1) 
        
        Li = l_ind;
        Lj = sub2ind(search_grid_dim, x_ind, y_ind-1, theta_ind, s_ind);
        
        % compute corresponding distance
        Tij = T_head_torso(:, Li);
        Tji = T_torso_head(:, Lj);
        dij = sum(abs(Tij - Tji));
        head_neighbors(3) = dij + head.B(x_ind, y_ind-1, theta_ind, s_ind);
        
        Tij = T_upper_arm_r_torso(:, Li);
        Tji = T_torso_upper_arm_r(:, Lj);
        dij = sum(abs(Tij - Tji));
        upper_arm_r_neighbors(3) = dij + upper_arm_r.B(x_ind, y_ind-1, theta_ind, s_ind);

        Tij = T_upper_arm_l_torso(:, Li);
        Tji = T_torso_upper_arm_l(:, Lj);
        dij = sum(abs(Tij - Tji));
        upper_arm_l_neighbors(3) = dij + upper_arm_l.B(x_ind, y_ind-1, theta_ind, s_ind);
        
        % take down the lj that gives this distance
        head_neighbors_lj(3, :) = head.Bj_p{x_ind, y_ind-1, theta_ind, s_ind};
        upper_arm_r_neighbors_lj(3, :) = upper_arm_r.Bj_p{x_ind, y_ind-1, theta_ind, s_ind};
        upper_arm_l_neighbors_lj(3, :) = upper_arm_l.Bj_p{x_ind, y_ind-1, theta_ind, s_ind};
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % neighbor in theta-direction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (theta_ind-1 >= 1)
        
        Li = l_ind;
        Lj = sub2ind(search_grid_dim, x_ind, y_ind, theta_ind-1, s_ind);
        
        % compute corresponding distance
        Tij = T_head_torso(:, Li);
        Tji = T_torso_head(:, Lj);
        dij = sum(abs(Tij - Tji));
        head_neighbors(4) = dij + head.B(x_ind, y_ind, theta_ind-1, s_ind);
        
        Tij = T_upper_arm_r_torso(:, Li);
        Tji = T_torso_upper_arm_r(:, Lj);
        dij = sum(abs(Tij - Tji));
        upper_arm_r_neighbors(4) = dij + upper_arm_r.B(x_ind, y_ind, theta_ind-1, s_ind);

        Tij = T_upper_arm_l_torso(:, Li);
        Tji = T_torso_upper_arm_l(:, Lj);
        dij = sum(abs(Tij - Tji));
        upper_arm_l_neighbors(4) = dij + upper_arm_l.B(x_ind, y_ind, theta_ind-1, s_ind);
        
        % take down the lj that gives this distance
        head_neighbors_lj(4, :) = head.Bj_p{x_ind, y_ind, theta_ind-1, s_ind};
        upper_arm_r_neighbors_lj(4, :) = upper_arm_r.Bj_p{x_ind, y_ind, theta_ind-1, s_ind};
        upper_arm_l_neighbors_lj(4, :) = upper_arm_l.Bj_p{x_ind, y_ind, theta_ind-1, s_ind};
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % neighbor in s-direction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (s_ind-1 >= 1) 
        
        Li = l_ind;
        Lj = sub2ind(search_grid_dim, x_ind, y_ind, theta_ind, s_ind-1);
        
        % compute corresponding distance
        Tij = T_head_torso(:, Li);
        Tji = T_torso_head(:, Lj);
        dij = sum(abs(Tij - Tji));
        head_neighbors(5) = dij + head.B(x_ind, y_ind, theta_ind, s_ind-1);
        
        Tij = T_upper_arm_r_torso(:, Li);
        Tji = T_torso_upper_arm_r(:, Lj);
        dij = sum(abs(Tij - Tji));
        upper_arm_r_neighbors(5) = dij + upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind-1);

        Tij = T_upper_arm_l_torso(:, Li);
        Tji = T_torso_upper_arm_l(:, Lj);
        dij = sum(abs(Tij - Tji));
        upper_arm_l_neighbors(5) = dij + upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind-1);
        
        % take down the lj that gives this distance
        head_neighbors_lj(5, :) = head.Bj_p{x_ind, y_ind, theta_ind, s_ind-1};
        upper_arm_r_neighbors_lj(5, :) = upper_arm_r.Bj_p{x_ind, y_ind, theta_ind, s_ind-1};
        upper_arm_l_neighbors_lj(5, :) = upper_arm_l.Bj_p{x_ind, y_ind, theta_ind, s_ind-1};
    end

    % obtain the min/argmin value
    [head.B(x_ind, y_ind, theta_ind, s_ind), head_min_ind] = min(head_neighbors);
    [upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind), upper_arm_r_min_ind] = min(upper_arm_r_neighbors);
    [upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind), upper_arm_l_min_ind] = min(upper_arm_l_neighbors);

    head.Bj_p{x_ind, y_ind, theta_ind, s_ind} = head_neighbors_lj(head_min_ind, :);
    upper_arm_r.Bj_p{x_ind, y_ind, theta_ind, s_ind} = upper_arm_r_neighbors_lj(upper_arm_r_min_ind, :);
    upper_arm_l.Bj_p{x_ind, y_ind, theta_ind, s_ind} = upper_arm_l_neighbors_lj(upper_arm_l_min_ind, :);

end

%% Solve for optimal configuration for torso
fprintf('Solving for optimal torso configuration...\n');

torso_opt_L = [1, 1, 1, 1];
torso_opt_E = Inf;
for l_ind = 1 : n_search
                
    if (mod(l_ind, 10000) == 0)
        fprintf('Solving for optimal configuration of torso, progress: %.0f%%\n', 100*l_ind/size(search_grid, 2)); 
    end
    
    [x_ind, y_ind, theta_ind, s_ind] = ind2sub(search_grid_dim, l_ind);
    L = search_grid(:, l_ind)';
    
    head_energy = head.B(x_ind, y_ind, theta_ind, s_ind);
    upper_arm_r_energy = upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind);
    upper_arm_l_energy = upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind);

    % compute the total energy
    torso.B(x_ind, y_ind, theta_ind, s_ind) = ...
        match_energy_cost(L, torso.part_id, dat_pt(:,torso.part_id)) + ...
        head_energy + upper_arm_r_energy + upper_arm_l_energy; 

    % update the minimum value
    if (torso.B(x_ind, y_ind, theta_ind, s_ind) < torso_opt_E)
        torso_opt_E = torso.B(x_ind, y_ind, theta_ind, s_ind);
        torso_opt_L = [x_ind, y_ind, theta_ind, s_ind];
    end
end

%% Obtain the coordinate of each node
% torso
torso_x = x_grid(torso_opt_L(1));
torso_y = y_grid(torso_opt_L(2));
torso_theta = theta_grid(torso_opt_L(3));
torso_s = s_grid(torso_opt_L(4));
torso_corr = [torso_x - torso_s * opt.model.len(1)/2 * sin(torso_theta); ...
              torso_y - torso_s * opt.model.len(1)/2 * cos(torso_theta); ...
              torso_x + torso_s * opt.model.len(1)/2 * sin(torso_theta); ...
              torso_y + torso_s * opt.model.len(1)/2 * cos(torso_theta)];

% left-upper arm
upper_arm_l_corr = upper_arm_l.Bj_p{torso_opt_L(1), torso_opt_L(2), torso_opt_L(3), torso_opt_L(4)};
upper_arm_l_corr = [upper_arm_l_corr(1) - upper_arm_l_corr(4) * opt.model.len(2)/2 * cos(upper_arm_l_corr(3)); ...
                    upper_arm_l_corr(2) - upper_arm_l_corr(4) * opt.model.len(2)/2 * sin(upper_arm_l_corr(3)); ...
                    upper_arm_l_corr(1) + upper_arm_l_corr(4) * opt.model.len(2)/2 * cos(upper_arm_l_corr(3)); ...
                    upper_arm_l_corr(2) + upper_arm_l_corr(4) * opt.model.len(2)/2 * sin(upper_arm_l_corr(3))];

% right-upper arm
upper_arm_r_corr = upper_arm_r.Bj_p{torso_opt_L(1), torso_opt_L(2), torso_opt_L(3), torso_opt_L(4)};
upper_arm_r_corr = [upper_arm_r_corr(1) - upper_arm_r_corr(4) * opt.model.len(3)/2 * cos(upper_arm_r_corr(3)); ...
                    upper_arm_r_corr(2) - upper_arm_r_corr(4) * opt.model.len(3)/2 * sin(upper_arm_r_corr(3)); ...
                    upper_arm_r_corr(1) + upper_arm_r_corr(4) * opt.model.len(3)/2 * cos(upper_arm_r_corr(3)); ...
                    upper_arm_r_corr(2) + upper_arm_r_corr(4) * opt.model.len(3)/2 * sin(upper_arm_r_corr(3))];

% head
head_corr = head.Bj_p{torso_opt_L(1), torso_opt_L(2), torso_opt_L(3), torso_opt_L(4)};
head_corr = [head_corr(1) - head_corr(4) * opt.model.len(6)/2 * sin(head_corr(3)); ...
             head_corr(2) - head_corr(4) * opt.model.len(6)/2 * cos(head_corr(3)); ...
             head_corr(1) + head_corr(4) * opt.model.len(6)/2 * sin(head_corr(3)); ...
             head_corr(2) + head_corr(4) * opt.model.len(6)/2 * cos(head_corr(3))];

% 4 part corr
total_corr = [torso_corr upper_arm_l_corr upper_arm_r_corr head_corr];

% Draw stickman
colors = [0.99 0 0 0; 0 0.99 0.99 0; 0 0 0 0.99]; 
% torso - red; arms - green; head - blue
thickness = 4;
drawidx = true;
drawfullskeleton = 1;
hdl = DrawStickman(total_corr, img, colors, thickness, drawidx);