%% Test case after obtaining the optimal configuration of all parts
% consider head
head_corr = head.Bj_p{torso_opt_L(1), torso_opt_L(2), torso_opt_L(3), torso_opt_L(4)};
theta_ind = find(theta_grid == head_corr(3));
s_ind     = find(s_grid == head_corr(4));
head_B_xy = head.B(:, :, theta_ind, s_ind);

% calculate the gradient along each variable
[g_x, g_y, g_theta, g_s] = gradient(head.B);
figure;
subplot(2,2,1), hist(g_x(:)), title('g_x');
subplot(2,2,2), hist(g_y(:)), title('g_y');
subplot(2,2,3), hist(g_theta(:)), title('g_{theta}');
subplot(2,2,4), hist(g_s(:)), title('g_s');

%% Re-run forward pass and inspect dij

head_dij = zeros(5, n_search);
upper_arm_l_dij = zeros(5, n_search);
upper_arm_r_dij = zeros(5, n_search);

head_fw = zeros(5, n_search);
upper_arm_l_fw = zeros(5, n_search);
upper_arm_r_fw = zeros(5, n_search);

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
    head_dij(1,l_ind) = dij;
    head_fw(1,l_ind)  = head.B(x_ind, y_ind, theta_ind, s_ind);
    head_neighbors(1)  = dij + head.B(x_ind, y_ind, theta_ind, s_ind);

    Tij = T_upper_arm_l_torso(:, Li);
    Tji = T_torso_upper_arm_l(:, Lj);
    dij = sum(abs(Tij - Tji));
    upper_arm_l_dij(1,l_ind) = dij;
    upper_arm_l_fw(1,l_ind)  = head.B(x_ind, y_ind, theta_ind, s_ind);
    upper_arm_l_neighbors(1) = dij + upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind);

    Tij = T_upper_arm_r_torso(:, Li);
    Tji = T_torso_upper_arm_r(:, Lj);
    dij = sum(abs(Tij - Tji));
    upper_arm_r_dij(1,l_ind) = dij;
    upper_arm_r_fw(1,l_ind)  = head.B(x_ind, y_ind, theta_ind, s_ind);
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
        head_dij(2,l_ind) = dij;
        head_fw(2,l_ind)  = head.B(x_ind+1, y_ind, theta_ind, s_ind);
        head_neighbors(2) = dij + head.B(x_ind+1, y_ind, theta_ind, s_ind);
        
        Tij = T_upper_arm_r_torso(:, Li);
        Tji = T_torso_upper_arm_r(:, Lj);
        dij = sum(abs(Tij - Tji));
        upper_arm_r_dij(2,l_ind) = dij;
        upper_arm_r_fw(2,l_ind)  = head.B(x_ind+1, y_ind, theta_ind, s_ind);
        upper_arm_r_neighbors(2) = dij + upper_arm_r.B(x_ind+1, y_ind, theta_ind, s_ind);

        Tij = T_upper_arm_l_torso(:, Li);
        Tji = T_torso_upper_arm_l(:, Lj);
        dij = sum(abs(Tij - Tji));
        upper_arm_l_dij(2,l_ind) = dij;
        upper_arm_l_fw(2,l_ind)  = head.B(x_ind+1, y_ind, theta_ind, s_ind);
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
        head_dij(3,l_ind) = dij;
        head_fw(3,l_ind)  = head.B(x_ind, y_ind+1, theta_ind, s_ind);        
        head_neighbors(3) = dij + head.B(x_ind, y_ind+1, theta_ind, s_ind);
        
        Tij = T_upper_arm_r_torso(:, Li);
        Tji = T_torso_upper_arm_r(:, Lj);
        dij = sum(abs(Tij - Tji));
        upper_arm_r_dij(3,l_ind) = dij;
        upper_arm_r_fw(3,l_ind)  = head.B(x_ind, y_ind+1, theta_ind, s_ind);
        upper_arm_r_neighbors(3) = dij + upper_arm_r.B(x_ind, y_ind+1, theta_ind, s_ind);

        Tij = T_upper_arm_l_torso(:, Li);
        Tji = T_torso_upper_arm_l(:, Lj);
        dij = sum(abs(Tij - Tji));
        upper_arm_l_dij(3,l_ind) = dij;
        upper_arm_l_fw(3,l_ind)  = head.B(x_ind, y_ind+1, theta_ind, s_ind);
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
        head_dij(4,l_ind) = dij;
        head_fw(4,l_ind)  = head.B(x_ind, y_ind, theta_ind+1, s_ind);        
        head_neighbors(4) = dij + head.B(x_ind, y_ind, theta_ind+1, s_ind);
        
        Tij = T_upper_arm_r_torso(:, Li);
        Tji = T_torso_upper_arm_r(:, Lj);
        dij = sum(abs(Tij - Tji));
        upper_arm_r_dij(4,l_ind) = dij;
        upper_arm_r_fw(4,l_ind)  = head.B(x_ind, y_ind, theta_ind+1, s_ind);
        upper_arm_r_neighbors(4) = dij + upper_arm_r.B(x_ind, y_ind, theta_ind+1, s_ind);

        Tij = T_upper_arm_l_torso(:, Li);
        Tji = T_torso_upper_arm_l(:, Lj);
        dij = sum(abs(Tij - Tji));
        upper_arm_l_dij(4,l_ind) = dij;
        upper_arm_l_fw(4,l_ind)  = head.B(x_ind, y_ind, theta_ind+1, s_ind);
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
        head_dij(5,l_ind) = dij;
        head_fw(5,l_ind)  = head.B(x_ind, y_ind, theta_ind, s_ind+1);       
        head_neighbors(5) = dij + head.B(x_ind, y_ind, theta_ind, s_ind+1);
        
        Tij = T_upper_arm_r_torso(:, Li);
        Tji = T_torso_upper_arm_r(:, Lj);
        dij = sum(abs(Tij - Tji));
        upper_arm_r_dij(5,l_ind) = dij;
        upper_arm_r_fw(5,l_ind)  = head.B(x_ind, y_ind, theta_ind, s_ind+1);
        upper_arm_r_neighbors(5) = dij + upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind+1);

        Tij = T_upper_arm_l_torso(:, Li);
        Tji = T_torso_upper_arm_l(:, Lj);
        dij = sum(abs(Tij - Tji));
        upper_arm_l_dij(5,l_ind) = dij;
        upper_arm_l_fw(5,l_ind)  = head.B(x_ind, y_ind, theta_ind, s_ind+1);
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

%% visulization

% head
figure;
plot(head_dij(1,:)), hold on;
%plot(head_Dij(2,:)), hold on;
% plot(head_Dij(3,:)), hold on;
% plot(head_Dij(4,:)), hold on;
% plot(head_Dij(5,:)), hold off;

% upper_arm_l
%figure;
%plot(upper_arm_l_Dij(1,:)), hold on;
%plot(upper_arm_l_Dij(2,:)), hold on;
%plot(upper_arm_l_Dij(3,:)), hold on;
%plot(upper_arm_l_Dij(4,:)), hold on;

% upper_arm_r
%figure;
%plot(upper_arm_r_Dij(1,:)), hold on;
%plot(upper_arm_r_Dij(2,:)), hold on;
%plot(upper_arm_r_Dij(3,:)), hold on;
%plot(upper_arm_r_Dij(4,:)), hold on;