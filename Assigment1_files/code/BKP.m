% for x_ind = 1 : length(x_grid)
%     
%     % display progress
%     fprintf('Progress: %.0f%%\n', 100*x_ind/length(x_grid)); 
%     
%     for y_ind = 1 : length(y_grid)
%         for theta_ind = 1 : length(theta_grid)
%             for s_ind = 1 : length(s_grid)
%                 
%                 % retrieve the coordinate
%                 x = x_grid(x_ind);
%                 y = y_grid(y_ind);
%                 s = s_grid(s_ind);
%                 theta = theta_grid(theta_ind);
%                 
%                 L = [x, y, theta, s];
%                 
%                 % compute the matching cost
%                 head.B(x_ind, y_ind, theta_ind, s_ind) = match_energy_cost(L, head.part_id, dat_pt(:,head.part_id));
%                 upper_arm_r.B(x_ind, y_ind, theta_ind, s_ind) = match_energy_cost(L, upper_arm_r.part_id, dat_pt(:,upper_arm_r.part_id));
%                 upper_arm_l.B(x_ind, y_ind, theta_ind, s_ind) = match_energy_cost(L, upper_arm_l.part_id, dat_pt(:,upper_arm_l.part_id));                
%             
%                 % initialize the Bj_p
%                 head.Bj_p{x_ind, y_ind, theta_ind, s_ind} = L;
%                 upper_arm_r.Bj_p{x_ind, y_ind, theta_ind, s_ind} = L;
%                 upper_arm_l.Bj_p{x_ind, y_ind, theta_ind, s_ind} = L;
%             end
%         end
%     end
% end

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

