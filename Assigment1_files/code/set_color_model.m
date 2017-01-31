%% set up path for image samples
negative_path = fullfile(pwd, 'part_samples', 'Negatives_Resized');
positive_path = fullfile(pwd, 'part_samples', 'Positives_Resized');

%% Read in all image samples
% face samples
fprintf('Reading in positive samples...\n');
face_samples = dir(fullfile(positive_path, '*.jpg'));      
nfiles = length(face_samples);
face_images = cell(1, nfiles);

for i = 1 : nfiles
   filename = fullfile(positive_path, face_samples(i).name);
   im = im2double(imread(filename));
   face_images{i} = im;
end

% % negative samples
% fprintf('Reading in negative samples...\n');
% negative_samples = dir(fullfile(negative_path, '*.jpg'));      
% nfiles = length(negative_samples);
% negative_images = cell(1, nfiles);
% 
% for i = 1 : nfiles
%    filename = fullfile(negative_path, negative_samples(i).name);
%    im = imread(filename);
%    negative_images{i} = im;
% end

%% Extract the skin color
% calculate the mean image
fprintf('Solving for mean and covariance of color...\n');
[m, n, p] = size(face_images{1});
im = zeros(m, n, p);
for i = 1 : nfiles
    im = im + face_images{i};
end
im_mean = im/nfiles;

% get the skin color
mean_skin_color = squeeze(mean(mean(im_mean, 2), 1));

%% Compute the variance of skin color
A = reshape(im_mean, m*n, 3);
cov_skin_color = cov(A);