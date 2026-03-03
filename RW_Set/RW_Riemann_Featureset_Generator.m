%% Riemannian Feature Set Generator Script
% Original Author: JoeyBG. 
% Improved By: JoeyBG. 
% Date: 2025-12-05. 
% Affiliate: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description: 
%   1. Reads triplets of RTM, DTM, and RDM images sequentially.
%   2. Resizes images to 1024x1024.
%   3. Extracts Riemannian features for all three domains.
%   4. Constructs a 3-channel RGB image:
%       - Channel R: RTM Feature Map
%       - Channel G: DTM Feature Map
%       - Channel B: RDM Feature Map
%   5. Saves as .png images in 'RW_Feature_Set' with detailed logging.

%% Initialization of MATLAB Script
clear all;
close all;
clc;
disp('---------- © Author: JoeyBG © ----------');

%% Configuration and Parameters
% Directory settings
Dir_RTM = "RW_RTM_Set";
Dir_DTM = "RW_DTM_Set";
Dir_RDM = "RW_RDM_Set";
Dir_Out = "RW_Feature_Set";

% Sub-folder list
Sub_Folders = [...
    "P1_Gun", "P1_Nogun", ...
    "P2_Gun", "P2_Nogun", ...
    "P3_Gun", "P3_Nogun", ...
    "P4_Gun", "P4_Nogun"];

% Target size
Target_Size = [1024, 1024];

% Riemannian feature parameters
Para.Window_Size = 5;
Para.Regularization = 1e-6;
Para.Max_Iter_Mean = 10;
Para.Tol_Mean = 1e-4;
Para.Scales = [3, 5, 7];
Para.Use_Multiscale = true; 
Para.Max_Samples_Mean = 1024;

%% 2. Main Processing Loop
fprintf('Riemannian Feature Set Generator (RTM/DTM/RDM)\n');

total_folders = length(Sub_Folders);

for f = 1:total_folders
    curr_sub = Sub_Folders(f);
    path_rtm_sub = fullfile(Dir_RTM, curr_sub);
    path_dtm_sub = fullfile(Dir_DTM, curr_sub);
    path_rdm_sub = fullfile(Dir_RDM, curr_sub);
    path_out_sub = fullfile(Dir_Out, curr_sub);
    
    % Check input existence
    if ~isfolder(path_rtm_sub) || ~isfolder(path_dtm_sub) || ~isfolder(path_rdm_sub)
        warning('Folder triplet missing for: %s. Skipping.', curr_sub);
        continue;
    end
    
    % Create output folder
    if ~isfolder(path_out_sub)
        mkdir(path_out_sub);
    end
    
    % Get image list
    img_list = dir(fullfile(path_rtm_sub, '*.png')); 
    if isempty(img_list)
        img_list = dir(fullfile(path_rtm_sub, '*.jpg')); 
    end
    
    num_imgs = length(img_list);
    fprintf('\n--- Entering Folder: %s (%d/%d) ---\n', curr_sub, f, total_folders);
    fprintf('Found %d image triplets.\n', num_imgs);
    
    % Sequential loop
    for i = 1:num_imgs 
        
        file_name = img_list(i).name;
        full_path_rtm = fullfile(path_rtm_sub, file_name);
        full_path_dtm = fullfile(path_dtm_sub, file_name);
        full_path_rdm = fullfile(path_rdm_sub, file_name);
        
        % Start console log for this file
        fprintf('  > [%03d/%03d] Processing: %-25s ... ', i, num_imgs, file_name);
        
        % Check DTM and RDM existence
        if ~isfile(full_path_dtm)
            fprintf('[SKIPPED] Missing DTM file.\n');
            continue;
        end
        if ~isfile(full_path_rdm)
            fprintf('[SKIPPED] Missing RDM file.\n');
            continue;
        end
        
        try
            % --- Step A: Load and preprocess ---
            img_RTM = imread(full_path_rtm);
            img_DTM = imread(full_path_dtm);
            img_RDM = imread(full_path_rdm);
            
            % Convert to grayscale if RGB
            if size(img_RTM, 3) == 3, img_RTM = rgb2gray(img_RTM); end
            if size(img_DTM, 3) == 3, img_DTM = rgb2gray(img_DTM); end
            if size(img_RDM, 3) == 3, img_RDM = rgb2gray(img_RDM); end
            
            % Resize to 1024x1024
            img_RTM = imresize(img_RTM, Target_Size);
            img_DTM = imresize(img_DTM, Target_Size);
            img_RDM = imresize(img_RDM, Target_Size);
            
            % Convert to double [0, 1]
            img_RTM_dbl = im2double(img_RTM);
            img_DTM_dbl = im2double(img_DTM);
            img_RDM_dbl = im2double(img_RDM);
            
            % --- Step B: Feature extraction ---
            % Extract features for all three domains
            feat_RTM = extractRiemannianFrobenius_Silent(img_RTM_dbl, Para);
            feat_DTM = extractRiemannianFrobenius_Silent(img_DTM_dbl, Para);
            feat_RDM = extractRiemannianFrobenius_Silent(img_RDM_dbl, Para);
            
            % --- Step C: RGB construction ---
            % Channel 1 (Red)   = RTM Feature
            % Channel 2 (Green) = DTM Feature
            % Channel 3 (Blue)  = RDM Feature
            RGB_Feature = cat(3, feat_RTM, feat_DTM, feat_RDM);
            
            % --- Step D: Save as PNG ---
            [~, name_core, ~] = fileparts(file_name);
            save_name = fullfile(path_out_sub, [name_core, '.png']);
            
            imwrite(RGB_Feature, save_name);
            
            % End console log for this file
            fprintf('[SUCCESS]\n');
            
        catch ME
            fprintf('[FAILED]\n');
            fprintf('    Error Detail: %s\n', ME.message);
        end
    end
end

fprintf('All Processing Completed! Dataset saved in "%s".\n', Dir_Out);

%% Helper Functions
function feature_map = extractRiemannianFrobenius_Silent(image, P)
    [H, W] = size(image);

    [Ix, Iy] = imgradientxy(image, 'sobel');
    [Ixx, ~] = imgradientxy(Ix, 'sobel');
    [~, Iyy] = imgradientxy(Iy, 'sobel');
    grad_mag = sqrt(Ix.^2 + Iy.^2);

    if P.Use_Multiscale
        cov_field = computeMultiscaleCovField_Silent(image, Ix, Iy, Ixx, Iyy, grad_mag, P.Scales, P.Regularization);
    else
        cov_field = computeCovField_Silent(image, Ix, Iy, Ixx, Iyy, grad_mag, P.Window_Size, P.Regularization);
    end

    global_mean = computeRiemannianMean_Silent(cov_field, P.Max_Iter_Mean, P.Tol_Mean, P.Max_Samples_Mean);
    feature_map = zeros(H, W);
    [V, D] = eig(global_mean);
    D = diag(max(diag(D), 1e-10));
    inv_sqrt_M = V * diag(1./sqrt(diag(D))) * V';
    
    for i = 1:H
        for j = 1:W
            C = cov_field{i, j};

            M_temp = inv_sqrt_M * C * inv_sqrt_M;
            M_temp = (M_temp + M_temp') / 2;
            [Vm, Dm] = eig(M_temp);
            log_vals = log(max(diag(Dm), 1e-10));

            feature_map(i, j) = norm(log_vals); 
        end
    end

    min_v = min(feature_map(:));
    max_v = max(feature_map(:));
    if max_v > min_v
        feature_map = (feature_map - min_v) / (max_v - min_v);
    else
        feature_map = zeros(size(feature_map));
    end
end

function cov_field = computeCovField_Silent(image, Ix, Iy, Ixx, Iyy, grad_mag, win_size, reg)
    [H, W] = size(image);
    half_win = floor(win_size / 2);
    cov_field = cell(H, W);
    
    pad_style = 'symmetric';
    pad_sz = [half_win, half_win];
    img_p = padarray(image, pad_sz, pad_style);
    Ix_p = padarray(Ix, pad_sz, pad_style);
    Iy_p = padarray(Iy, pad_sz, pad_style);
    Ixx_p = padarray(Ixx, pad_sz, pad_style);
    Iyy_p = padarray(Iyy, pad_sz, pad_style);
    gm_p = padarray(grad_mag, pad_sz, pad_style);
    
    [xx, yy] = meshgrid(1:win_size, 1:win_size);
    xx = (xx(:) - half_win - 1) / half_win;
    yy = (yy(:) - half_win - 1) / half_win;
    
    for i = 1:H
        for j = 1:W
            r = i:(i + 2*half_win);
            c = j:(j + 2*half_win);
            
            F = [xx, yy, ...
                 reshape(img_p(r,c),[],1), ...
                 reshape(Ix_p(r,c),[],1), reshape(Iy_p(r,c),[],1), ...
                 reshape(Ixx_p(r,c),[],1), reshape(Iyy_p(r,c),[],1), ...
                 reshape(gm_p(r,c),[],1)];
             
            F = F - mean(F, 1);
            C = (F' * F) / (size(F, 1) - 1);
            cov_field{i, j} = C + reg * eye(size(C));
        end
    end
end

function cov_field = computeMultiscaleCovField_Silent(image, Ix, Iy, Ixx, Iyy, grad_mag, scales, reg)
    [H, W] = size(image);
    num_scales = length(scales);
    fields = cell(1, num_scales);
    for s = 1:num_scales
        fields{s} = computeCovField_Silent(image, Ix, Iy, Ixx, Iyy, grad_mag, scales(s), reg);
    end
    cov_field = cell(H, W);
    for i = 1:H
        for j = 1:W
            sum_log = zeros(size(fields{1}{1,1}));
            for s = 1:num_scales
                C = fields{s}{i, j};
                [V, D] = eig(C);
                sum_log = sum_log + V * diag(log(max(diag(D), 1e-10))) * V';
            end
            mean_log = sum_log / num_scales;
            [V, D] = eig(mean_log);
            cov_field{i, j} = V * diag(exp(diag(D))) * V';
        end
    end
end

function mean_C = computeRiemannianMean_Silent(cov_field, max_iter, tol, max_samples)
    [H, W] = size(cov_field);
    N_total = H * W;
    if N_total > max_samples
        indices = randperm(N_total, max_samples);
    else
        indices = 1:N_total;
    end
    matrices = cell(1, length(indices));
    for k = 1:length(indices)
        [r, c] = ind2sub([H, W], indices(k));
        matrices{k} = cov_field{r, c};
    end
    N = length(matrices);

    mean_C = matrices{1}; 
    for k = 2:N, mean_C = mean_C + matrices{k}; end
    mean_C = mean_C / N;

    for iter = 1:max_iter
        [V, D] = eig(mean_C);
        D = diag(max(diag(D), 1e-10));
        inv_sqrt_C = V * diag(1./sqrt(diag(D))) * V';
        sqrt_C = V * diag(sqrt(diag(D))) * V';
        
        tangent_sum = zeros(size(mean_C));
        for k = 1:N
            M = inv_sqrt_C * matrices{k} * inv_sqrt_C;
            M = (M + M') / 2;
            [Vm, Dm] = eig(M);
            log_M = Vm * diag(log(max(diag(Dm), 1e-10))) * Vm';
            tangent_sum = tangent_sum + log_M;
        end
        tangent_mean = tangent_sum / N;
        
        if norm(tangent_mean, 'fro') < tol
            break;
        end
        
        [Vt, Dt] = eig(tangent_mean);
        exp_tangent = Vt * diag(exp(diag(Dt))) * Vt';
        mean_C = sqrt_C * exp_tangent * sqrt_C;
        mean_C = (mean_C + mean_C') / 2;
    end
end