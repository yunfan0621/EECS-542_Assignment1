clc; close all; clear all;

% Initialization

fprintf('Initializing...\n');

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

% Startup command
installmex;
startup;
set_opt;

% Test calculate Tij
% x_grid = randn(1, 2);
% y_grid = randn(1, 2);
% theta_grid = randn(1,2);
% s_grid = randn(1, 2);
search_grid_dim = [2, 2, 2, 2];
x_grid = [1, 2];
y_grid = [1, 2];
theta_grid = [1, 2];
s_grid = [1, 2];
total_size = size(x_grid, 2) * size(y_grid, 2) * size(theta_grid, 2) * size(s_grid, 2);

i = 1;
j = 6;

Tij = calc_Tij(i, j, x_grid, y_grid, theta_grid, s_grid, opt);
Tij_test = zeros(4, total_size);
k = 1;
tic;
for s_ind = 1:length(s_grid)
    for theta_ind = 1:length(theta_grid)
        for y_ind = 1:length(y_grid)
            for x_ind = 1:length(x_grid)
                x = x_grid(x_ind);
                y = y_grid(y_ind);
                theta = theta_grid(theta_ind);
                s = s_grid(s_ind);
                
                Tij_test(3, k) = opt.wij.theta * (theta - opt.model.theta_ij(i, j)/2);
                Tij_test(4, k) = opt.wij.s * (log(s) - log(opt.model.s_ij(i, j))/2);
                W_ij = diag([opt.wij.x, opt.wij.y]);
                new_xy = W_ij * ([x, y]' + s * [cos(theta), -sin(theta); sin(theta), cos(theta)] * [opt.model.x_ij(i, j), opt.model.y_ij(i, j)]');
                
                Tij_test(1, k) = new_xy(1);
                Tij_test(2, k) = new_xy(2);
                % update k
                k = k + 1;
            end
        end
    end
end
toc;
D = (Tij - Tij_test).^2;
error = sum(D(:));
if (error < 1e-6)
    fprintf('PASS TEST Calculate Tij');
else
    fprintf('FAIL TEST Calculate Tij');
    exit;
end







