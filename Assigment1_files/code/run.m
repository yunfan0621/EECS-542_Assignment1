clc; clear all;

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
               'Bj_p',        []); % root node

head  = struct('name',        'head',...
               'part_id',     6,...
               'B',           [],...
               'Bj_p',        []); % leaf node

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
torso.B = zeros(1, n_search);
head.B  = zeros(size(torso.B));

% Bj_p(li) = argmin(D(Tij(li))), the lj associated with the minimum
% distance (propagated)
torso.Bj_p = cell(1, n_search);
head.Bj_p  = cell(size(torso.B));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pre-compute Tij and Tji for acceleration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T_head_torso = calc_Tij(head.part_id, torso.part_id, x_grid, y_grid, theta_grid, s_grid, opt);
T_torso_head = calc_Tij(torso.part_id, head.part_id, x_grid, y_grid, theta_grid, s_grid, opt);

%% Compute f(w) for distance transformation and initialize D for leave nodes

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initializing D using f(w) for all leave node
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Initializing D using f(w) for all leave nodes...\n');

for l_ind = 1 : n_search % search for lj, each location of 'this' part
    
    Lj = search_grid(:, l_ind)';
    torso.B(:, l_ind) = match_energy_cost(Lj, torso.part_id, dat_pt(:, torso.part_id));
     head.B(:, l_ind) = match_energy_cost(Lj, head.part_id, dat_pt(:, head.part_id));
    
    head.Bj_p{:, l_ind} = Lj;
end

%% Testcase for f(w) initialization
% B = torso.B;
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
head_dij = zeros(5, n_search);
head_fw = zeros(5, n_search);

fprintf('\nForward pass through D for all leave nodes...\n');
for l_ind = 1 : n_search % seach for all possibility of li

    if (mod(l_ind, 5000) == 0)
        fprintf('Leave Node Forward Pass Progress: %.0f%%\n', 100*l_ind/size(search_grid, 2)); 
    end

    % calculate linear lindex
    [x_ind, y_ind, theta_ind, s_ind] = ind2sub(search_grid_dim, l_ind);
    Li = l_ind;
    Lj = Li;
    
    % initialize neighbor vectors
    head_neighbors = Inf(1, 5); % for min value
    head_neighbors_lj = zeros(5, 4); % for argmin value, 5-neighbors
   
    %%%%%%%%%%%
    % Lj = Li
    %%%%%%%%%%%
    Tij = T_head_torso(:, Li);
    Tji = T_torso_head(:, Lj);
    dij = sum(abs(Tij - Tji));
    head_dij(1,l_ind) = dij;
    head_fw(1,l_ind)  = head.B(Lj);
    head_neighbors(1)  = dij + opt.fw.weight * head.B(Lj);
    
    % take down the lj that gives this distance
    head_neighbors_lj(1, :) = head.Bj_p{Lj};

    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % neighbor in x-direction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (x_ind+1 <= opt.scan_nsample.x)
        Lj_x = sub2ind(search_grid_dim, x_ind+1, y_ind, theta_ind, s_ind);
        
        Tij = T_head_torso(:, Li);
        Tji = T_torso_head(:, Lj_x);
        dij = sum(abs(Tij - Tji));
        head_dij(2,l_ind) = dij;
        head_fw(2,l_ind)  = head.B(Lj_x);
        head_neighbors(2) = dij + opt.fw.weight * head.B(Lj_x) + opt.k.x;
        
        % take down the lj that gives this distance
        head_neighbors_lj(2, :) = head.Bj_p{Lj_x};
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % neighbor in y-direction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (y_ind+1 <= opt.scan_nsample.y) 
        Lj_y = sub2ind(search_grid_dim, x_ind, y_ind+1, theta_ind, s_ind);
        
        Tij = T_head_torso(:, Li);
        Tji = T_torso_head(:, Lj_y);
        dij = sum(abs(Tij - Tji));
        head_dij(3,l_ind) = dij;
        head_fw(3,l_ind)  = head.B(Lj_y);        
        head_neighbors(3) = dij + opt.fw.weight * head.B(Lj_y) + opt.k.y;
        
        % take down the lj that gives this distance
        head_neighbors_lj(3, :) = head.Bj_p{Lj_y};
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % neighbor in theta-direction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (theta_ind+1 <= opt.scan_nsample.theta)
        Lj_theta = sub2ind(search_grid_dim, x_ind, y_ind, theta_ind+1, s_ind);
        
        Tij = T_head_torso(:, Li);
        Tji = T_torso_head(:, Lj_theta);
        dij = sum(abs(Tij - Tji));
        head_dij(4,l_ind) = dij;
        head_fw(4,l_ind)  = head.B(Lj_theta);        
        head_neighbors(4) = dij + opt.fw.weight * head.B(Lj_theta) + opt.k.theta;
        
        % take down the lj that gives this distance
        head_neighbors_lj(4, :) = head.Bj_p{Lj_theta};
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % neighbor in s-direction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (s_ind+1 <= opt.scan_nsample.s) 
        Lj_s = sub2ind(search_grid_dim, x_ind, y_ind, theta_ind, s_ind+1);
        
        Tij = T_head_torso(:, Li);
        Tji = T_torso_head(:, Lj_s);
        dij = sum(abs(Tij - Tji));
        head_dij(5,l_ind) = dij;
        head_fw(5,l_ind)  = head.B(Lj_s);       
        head_neighbors(5) = dij + opt.fw.weight * head.B(Lj_s) + opt.k.s;
        
        % take down the lj that gives this distance
        head_neighbors_lj(5, :) = head.Bj_p{Lj_s};
    end

    % obtain the min/argmin value
    [head.B(Li), head_min_ind] = min(head_neighbors);
    head.Bj_p{Li} = head_neighbors_lj(head_min_ind, :);
end

%% Compute minimum distance in D for leave nodes (Backward Pass)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Backward pass through D to find the minimum value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nBackward pass through D for all leave nodes...\n');
for l_ind = n_search : -1 : 1    

    if (mod(l_ind, 5000) == 0)
        fprintf('Leave Node Backward Pass Progress: %.0f%%\n', 100*(n_search-l_ind+5000)/size(search_grid, 2)); 
    end

    % calculate linear lindex
    [x_ind, y_ind, theta_ind, s_ind] = ind2sub(search_grid_dim, l_ind);
    Li = l_ind;
    Lj = Li;  
    
    % initialize neighbor vectors
    head_neighbors = Inf(1, 5); % for min value
    head_neighbors_lj = zeros(5, 4); % for argmin value, 5-neighbors
    
    %%%%%%%%%%%
    % Lj = Li
    %%%%%%%%%%%

    Tij = T_head_torso(:, Li);
    Tji = T_torso_head(:, Lj);
    dij = sum(abs(Tij - Tji));
    head_neighbors(1) = dij + head.B(Lj);
    
    % take down the lj that gives this distance
    head_neighbors_lj(1, :) = head.Bj_p{Lj};

    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % neighbor in x-direction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (x_ind-1 >= 1)
        Lj_x = sub2ind(search_grid_dim, x_ind-1, y_ind, theta_ind, s_ind);
        
        Tij = T_head_torso(:, Li);
        Tji = T_torso_head(:, Lj_x);
        dij = sum(abs(Tij - Tji));
        head_neighbors(2) = dij + head.B(Lj_x) + opt.k.x;
        
        % take down the lj that gives this distance
        head_neighbors_lj(2, :) = head.Bj_p{Lj_x};
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % neighbor in y-direction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (y_ind-1 >= 1) 
        Lj_y = sub2ind(search_grid_dim, x_ind, y_ind-1, theta_ind, s_ind);
        
        Tij = T_head_torso(:, Li);
        Tji = T_torso_head(:, Lj_y);
        dij = sum(abs(Tij - Tji));
        head_neighbors(3) = dij + head.B(Lj_y) + opt.k.y;
        
        % take down the lj that gives this distance
        head_neighbors_lj(3, :) = head.Bj_p{Lj_y};
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % neighbor in theta-direction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (theta_ind-1 >= 1)
        Lj_theta = sub2ind(search_grid_dim, x_ind, y_ind, theta_ind-1, s_ind);
        
        Tij = T_head_torso(:, Li);
        Tji = T_torso_head(:, Lj_theta);
        dij = sum(abs(Tij - Tji));
        head_neighbors(4) = dij + head.B(Lj_theta) + opt.k.theta;
        
        % take down the lj that gives this distance
        head_neighbors_lj(4, :) = head.Bj_p{Lj_theta};
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % neighbor in s-direction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (s_ind-1 >= 1) 
        Lj_s = sub2ind(search_grid_dim, x_ind, y_ind, theta_ind, s_ind-1);
        
        Tij = T_head_torso(:, Li);
        Tji = T_torso_head(:, Lj_s);
        dij = sum(abs(Tij - Tji));
        head_neighbors(5) = dij + head.B(Lj_s) + opt.k.s;
        
        % take down the lj that gives this distance
        head_neighbors_lj(5, :) = head.Bj_p{Lj_s};
    end

    % obtain the min/argmin value
    [head.B(Li), head_min_ind] = min(head_neighbors);
    head.Bj_p{Li} = head_neighbors_lj(head_min_ind, :);
end

%% Testcase for f(w) initialization
B = head.B;
[val, ind] = min(B(:));
[x_ind, y_ind, theta_ind, s_ind] = ind2sub(search_grid_dim, ind);
fprintf('Optimal x: %d\n', x_grid(x_ind));
fprintf('Optimal y: %d\n', y_grid(y_ind));
fprintf('Optimal theta: %d\n', theta_grid(theta_ind));
fprintf('Optimal s: %d\n', s_grid(s_ind));

%% Solve for optimal configuration for torso
fprintf('Solving for optimal torso configuration...\n');
torso_opt_E = Inf;
for l_ind = 1 : n_search
                
    if (mod(l_ind, 50000) == 0)
        fprintf('Solving for optimal configuration of torso, progress: %.0f%%\n', 100*l_ind/size(search_grid, 2)); 
    end

    torso_energy = torso.B(l_ind);
    head_energy  = head.B(l_ind);

    % compute the total energy
    torso.B(l_ind) = torso_energy + head_energy; 

    % update the minimum value
    if (torso.B(l_ind) < torso_opt_E)
        torso_opt_E = torso.B(l_ind);
        [x_ind, y_ind, theta_ind, s_ind] = ind2sub(search_grid_dim, l_ind);
        torso_opt_L = [x_ind, y_ind, theta_ind, s_ind];
        torso_opt_ind = l_ind;
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

% head
head_corr = head.Bj_p{torso_opt_ind};
head_corr = [head_corr(1) - head_corr(4) * opt.model.len(6)/2 * sin(head_corr(3)); ...
             head_corr(2) - head_corr(4) * opt.model.len(6)/2 * cos(head_corr(3)); ...
             head_corr(1) + head_corr(4) * opt.model.len(6)/2 * sin(head_corr(3)); ...
             head_corr(2) + head_corr(4) * opt.model.len(6)/2 * cos(head_corr(3))];

% 4 part corr
total_corr = [torso_corr head_corr];

% Draw stickman
colors = [0.99 0; 0 0.99; 0 0]; 
% torso - red; arms - green; head - blue
thickness = 4;
drawidx = true;
drawfullskeleton = 1;
hdl = DrawStickman(total_corr, img, colors, thickness, drawidx);