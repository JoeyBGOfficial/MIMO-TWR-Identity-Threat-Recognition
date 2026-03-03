%% Riemannian Feature Set Generator Script GPU Accelerated Version
% Original Author: JoeyBG. 
% Optimized By: JoeyBG. 
% Date: 2025-12-05. 
% Affiliate: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description: 
%   1. Reads triplets of RTM, DTM, and RDM images sequentially.
%   2. Resizes images to 1024x1024.
%   3. Extracts Riemannian features for all three domains using GPU acceleration.
%   4. Constructs a 3-channel RGB image:
%       - Channel R: RTM Feature Map
%       - Channel G: DTM Feature Map
%       - Channel B: RDM Feature Map
%   5. Saves as .png images in RW_Feature_Set with detailed logging.

%% Initialization of MATLAB Script
clear all;
close all;
clc;
disp('---------- © Author: JoeyBG GPU Edition © ----------');

% Check GPU availability
if gpuDeviceCount < 1
    error('No GPU detected! This script requires an NVIDIA GPU and Parallel Computing Toolbox.');
else
    g = gpuDevice;
    fprintf('Using GPU: %s, Memory: %.2f GB\n', g.Name, g.AvailableMemory/1e9);
end

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
Para.Max_Samples_Mean = 2048;                                               % Increased for GPU batching efficiency

%% Main Processing Loop
fprintf('Riemannian Feature Set Generator RTM DTM RDM\n');

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
    fprintf('\n--- Entering Folder: %s %d of %d ---\n', curr_sub, f, total_folders);
    fprintf('Found %d image triplets.\n', num_imgs);
    
    % Sequential loop
    for i = 1:num_imgs 
        
        file_name = img_list(i).name;
        full_path_rtm = fullfile(path_rtm_sub, file_name);
        full_path_dtm = fullfile(path_dtm_sub, file_name);
        full_path_rdm = fullfile(path_rdm_sub, file_name);
        
        % Start console log for this file
        fprintf('  > %03d of %03d Processing: %-25s ... ', i, num_imgs, file_name);
        
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
            
            % Convert to double 0 to 1
            img_RTM_dbl = im2double(img_RTM);
            img_DTM_dbl = im2double(img_DTM);
            img_RDM_dbl = im2double(img_RDM);
            
            % --- Step B: Feature extraction ---
            % Extract features for all three domains
            feat_RTM = extractRiemannianFrobenius_GPU(img_RTM_dbl, Para);
            feat_DTM = extractRiemannianFrobenius_GPU(img_DTM_dbl, Para);
            feat_RDM = extractRiemannianFrobenius_GPU(img_RDM_dbl, Para);
            
            % --- Step C: RGB construction ---
            % Channel 1 Red is RTM Feature
            % Channel 2 Green is DTM Feature
            % Channel 3 Blue is RDM Feature
            RGB_Feature = cat(3, feat_RTM, feat_DTM, feat_RDM);
            
            % Step D: Save as PNG
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

%% Core Function: GPU Riemannian Frobenius Feature Extraction
function feature_map = extractRiemannianFrobenius_GPU(image, P)
    % Initialize GPU array
    image_gpu = gpuArray(image);
    [H, W] = size(image);
    
    % Compute gradients on GPU
    [Ix, Iy] = imgradientxy(image_gpu, 'sobel');
    [Ixx, ~] = imgradientxy(Ix, 'sobel');
    [~, Iyy] = imgradientxy(Iy, 'sobel');
    grad_mag = sqrt(Ix.^2 + Iy.^2);
    
    % Compute covariance matrix field
    if P.Use_Multiscale
        cov_data_all = computeMultiscaleCovField_GPU(image_gpu, Ix, Iy, Ixx, Iyy, ... 
            grad_mag, P.Scales, P.Regularization);
    else
        cov_data_all = computeCovField_GPU(image_gpu, Ix, Iy, Ixx, Iyy, grad_mag, ... 
            P.Window_Size, P.Regularization);
    end
    
    % Transfer data to CPU for complex algebra
    cov_field_cpu = gather(cov_data_all);
    clear cov_data_all; 
    
    % Reshape for processing
    num_pixels = H * W;
    
    % Compute global Riemannian mean
    rng(42);
    sample_indices = randperm(num_pixels, min(num_pixels, P.Max_Samples_Mean));
    sample_matrices = cov_field_cpu(:, :, sample_indices);
    
    % Convert to cell for compatibility
    N_samples = size(sample_matrices, 3);
    cell_samples = cell(1, N_samples);
    for k = 1:N_samples
        cell_samples{k} = sample_matrices(:,:,k);
    end
    
    global_mean = computeRiemannianMean_CPU(cell_samples, P.Max_Iter_Mean, P.Tol_Mean);
    
    % Extract Features
    feature_vec = zeros(num_pixels, 1);
    M = global_mean; 
    
    % Parallel Loop on CPU
    parfor k = 1:num_pixels
        % Extract single matrix
        C = cov_field_cpu(:, :, k);
        
        % Log map and Frobenius norm
        [V, D_eig] = eig(M);
        d_diag = diag(D_eig);
        d_diag = max(d_diag, 1e-10);
        inv_sqrt_M = V * diag(1./sqrt(d_diag)) * V';
        
        % Map C to tangent space of identity
        Sym = inv_sqrt_M * C * inv_sqrt_M;
        Sym = (Sym + Sym') / 2;
        
        [Vm, Dm] = eig(Sym);
        dm_diag = diag(Dm);
        dm_diag = max(dm_diag, 1e-10);
        
        % Log at identity
        Log_Sym = Vm * diag(log(dm_diag)) * Vm';        
        feature_vec(k) = norm(Log_Sym, 'fro');
    end
    
    % Reshape back to image
    feature_map = reshape(feature_vec, [H, W]);
    
    % Normalize
    min_v = min(feature_map(:));
    max_v = max(feature_map(:));
    if max_v > min_v
        feature_map = (feature_map - min_v) / (max_v - min_v);
    else
        feature_map = zeros(size(feature_map));
    end
end

%% Helper: Vectorized GPU Covariance Computation
function cov_stack = computeCovField_GPU(I, Ix, Iy, Ixx, Iyy, grad, win_size, reg)
    % Input: All images are gpuArrays
    % Output: Covariance stack is 8 by 8 by total pixels double    
    [H, W] = size(I);
    num_pixels = H*W;
    
    % Define kernels for convolution
    K_1 = ones(win_size, win_size, 'gpuArray'); 
    N = win_size^2;
    
    % Coordinate filters for local x and y covariance
    half = floor(win_size/2);
    [xx, yy] = meshgrid(-half:half, -half:half);
    K_x = gpuArray(xx / half); 
    K_y = gpuArray(yy / half);
    
    % Pre-compute channel list
    % Channels: 1:x, 2:y, 3:I, 4:Ix, 5:Iy, 6:Ixx, 7:Iyy, 8:grad    
    Channels = {[], [], I, Ix, Iy, Ixx, Iyy, grad};
    n_ch = 8;
    
    % Allocate output stack
    cov_stack = zeros(n_ch, n_ch, num_pixels, 'single'); 
    if canUseGPU() && gpuDevice().AvailableMemory > 8e9
        cov_stack = zeros(n_ch, n_ch, num_pixels, 'double');
    end
    
    % Helper for convolution
    do_conv = @(img, k) conv2(img, k, 'same');
    
    for r = 1:n_ch
        for c = r:n_ch           
            % Analyze Row
            if r == 1 % X coord
                if c == 1 % Cov x, x which is Constant
                   val = sum(sum(K_x .* K_x)); 
                   cov_val = val * ones(H, W, 'gpuArray');
                elseif c == 2 % Cov x, y which is Constant
                   val = sum(sum(K_x .* K_y));
                   cov_val = val * ones(H, W, 'gpuArray');
                else % Cov x, ImageChannel
                   cov_val = do_conv(Channels{c}, rot90(K_x,2)); 
                end
            elseif r == 2 % Y coord
                 if c == 2 % Cov y, y
                    val = sum(sum(K_y .* K_y));
                    cov_val = val * ones(H, W, 'gpuArray');
                 else % Cov y, ImageChannel
                    cov_val = do_conv(Channels{c}, rot90(K_y,2));
                 end
            else % Image vs Image
                 Sum_XY = do_conv(Channels{r} .* Channels{c}, K_1);
                 Sum_X  = do_conv(Channels{r}, K_1);
                 Sum_Y  = do_conv(Channels{c}, K_1);
                 
                 cov_val = Sum_XY - (Sum_X .* Sum_Y) / N;
            end
            
            % Unbiased normalization
            cov_val = cov_val / (N - 1);
            
            % Add regularization to diagonal
            if r == c
                cov_val = cov_val + reg;
            end
            
            % Store in stack
            reshaped_vals = reshape(cov_val, [1, 1, num_pixels]);
            cov_stack(r, c, :) = reshaped_vals;
            if r ~= c
                cov_stack(c, r, :) = reshaped_vals;
            end
        end
    end
end

%% Helper: Multi-Scale GPU Covariance
function cov_stack_final = computeMultiscaleCovField_GPU(I, Ix, Iy, Ixx, Iyy, ... 
    grad, scales, reg)
    
    [H, W] = size(I);
    num_pixels = H*W;
    n_ch = 8;
    
    % Accumulate matrices from different scales
    stack_accum = zeros(n_ch, n_ch, num_pixels, 'double'); 
    if canUseGPU(), stack_accum = gpuArray(stack_accum); end
    
    for s = 1:length(scales)
        current_stack = computeCovField_GPU(I, Ix, Iy, Ixx, Iyy, grad, scales(s), reg);
        stack_accum = stack_accum + current_stack;
    end
    
    % Simplification for speed: Arithmetic mean of covariances
    cov_stack_final = stack_accum / length(scales);
end

%% Helper: Riemannian Mean CPU
function mean_C = computeRiemannianMean_CPU(matrices, max_iter, tol)
    N = length(matrices);
    d = size(matrices{1}, 1);
    
    % Initialize with arithmetic mean
    mean_C = zeros(d);
    for k = 1:N, mean_C = mean_C + matrices{k}; end
    mean_C = mean_C / N;
    
    % Iteration
    for iter = 1:max_iter
        [V, D] = eig(mean_C);
        d_val = diag(D); d_val = max(d_val, 1e-10);
        sqrt_C = V * diag(sqrt(d_val)) * V';
        inv_sqrt_C = V * diag(1./sqrt(d_val)) * V';
        
        tangent_sum = zeros(d);
        
        % Sequential loop for small sample set
        for k = 1:N
            M = inv_sqrt_C * matrices{k} * inv_sqrt_C;
            M = (M + M') / 2;
            [Vm, Dm] = eig(M);
            dm_val = diag(Dm); dm_val = max(dm_val, 1e-10);
            log_M = Vm * diag(log(dm_val)) * Vm';
            tangent_sum = tangent_sum + log_M;
        end
        
        tangent_mean = tangent_sum / N;
        norm_val = norm(tangent_mean, 'fro');
        
        if norm_val < tol, break; end
        
        [Vt, Dt] = eig(tangent_mean);
        exp_tangent = Vt * diag(exp(diag(Dt))) * Vt';
        mean_C = sqrt_C * exp_tangent * sqrt_C;
        mean_C = (mean_C + mean_C') / 2;
    end
end

function b = canUseGPU()
    try
        g = gpuDevice;
        b = true;
    catch
        b = false;
    end
end