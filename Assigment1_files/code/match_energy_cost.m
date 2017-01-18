function cost= match_energy_cost(L, part, dat_pt)
% L --> [x, y, theta, scale]  end points of the query stick  
% part is the query part, 1=torso, 2=left upper arm, 3=right upper arm, 4=left lower arm, 5=right lower arm, 6= head
% using the model lengths as per the presentation

if part>1 && part <6
    % compute the error between L and each arm part
    % ???
    val1 = calc_val(dat_pt,L,2);
    val2 = calc_val(dat_pt,L,3);
    val3 = calc_val(dat_pt,L,4);
    val4 = calc_val(dat_pt,L,5);
    cost = min([val1, val2, val3, val4]);
else
    cost = calc_val(dat_pt,L,part);
end

end

function cost = calc_val(dat_pt, L, part)
% dat_pt: GT annotation (4x1 coor) for a certain part

model_len = [160, 95, 95, 65, 65, 60];

% calculate center and theta of GT annotation
dat_x = mean([dat_pt(1) dat_pt(3)]);
dat_y = mean([dat_pt(2) dat_pt(4)]);
dat_theta = atan((dat_pt(2)-dat_pt(4))/(dat_pt(1)-dat_pt(3)));

% special treatment of head and torso
if part==1 || part ==6 % ???
    if dat_theta < 0
        dat_theta = dat_theta+pi/2;
    else
        dat_theta = dat_theta-pi/2;
    end
end
% compute the scale (observed_len / model_len)
dat_scale = sqrt(sum([dat_pt(1)-dat_pt(3),dat_pt(2)-dat_pt(4)].^2))/model_len(part);
dat_val = [dat_x, dat_y, dat_theta, dat_scale]; % GT annotation grouped in [x, y, theta, scale manner]
diff = (dat_val-L).*[0.5 0.5 100 100];
cost = sum(diff.^2);

end