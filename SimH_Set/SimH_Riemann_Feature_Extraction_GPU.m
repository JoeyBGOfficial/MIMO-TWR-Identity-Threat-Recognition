%% Riemannian Manifold Feature Extraction Script: GPU Accelerated Version
% Original Author: JoeyBG. 
% Optimized By: JoeyBG. 
% Date: 2025-12-05. 
% Affiliate: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Introduction:
%   This is the High-Performance computing version of the Riemann processor.
%   Accelerations:
%       1. Covariance Construction: Replaced nested loops with GPU-based vectorized convolutions.
%       2. Manifold Operations: Utilizes CPU Parallel Pool for massive small-matrix algebra.
%   Speedup: Expected 20x to 100x faster than the CPU version depending on Image Size.
%
% Files to process:
%   RTM.png, DTM.png, RDM.png located in Example_Images folder. 

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

% Load configurations
try
    load("JoeyBG_Colormap.mat"); 
catch
    warning('JoeyBG_Colormap.mat is missed! Use jet instead.');
end

%% Parameter Definitions
% File and directory settings
Data_Folder = "Example_Images";
File_List = ["RTM.png", "DTM.png", "RDM.png"]; 

% Riemannian feature extraction parameters
Window_Size = 5;            
Regularization = 1e-6;      
Max_Iter_Mean = 15;         
Tol_Mean = 1e-6;            
Scales = [3, 5, 7];         
Use_Multiscale = true;      
Max_Samples_Mean = 2048;                                                    % Increased for GPU batching efficiency

% Visualization parameters
Vis_Params.Font_Name = 'Palatino Linotype';
Vis_Params.Font_Size_Basis = 15;
Vis_Params.Colormap = jet; 
if exist('CList_Flip', 'var'), Vis_Params.Colormap = CList_Flip; end

%% Main Processing Loop
fprintf('Starting GPU-Accelerated Riemannian Feature Extraction...\n');
total_files = length(File_List);

for f_idx = 1:total_files
    current_file_name = File_List(f_idx);
    full_path = fullfile(Data_Folder, current_file_name);
    [~, name_no_ext, ~] = fileparts(current_file_name);
    
    fprintf('Processing File %d of %d: %s\n', f_idx, total_files, current_file_name);
    
    try
        img_original = imread(full_path);
    catch
        warning('File %s not found. Skipping.', full_path);
        continue;
    end
    
    if size(img_original, 3) == 3, img_gray = rgb2gray(img_original); else, img_gray = img_original; end
    img_double = im2double(img_gray);
    img_double = (img_double - min(img_double(:))) / (max(img_double(:)) - min(img_double(:)));
    [H, W] = size(img_double);
    fprintf('  > Image size: %d x %d\n', H, W);
    
    % --- Start Timer ---
    tic;
    
    % Move data to GPU
    img_gpu = gpuArray(img_double);
    
    % Core processing
    feature_frobenius = extractRiemannianFrobenius_GPU(img_gpu, Window_Size, ... 
        Regularization, Max_Iter_Mean, Tol_Mean, Scales, Use_Multiscale, ... 
        Max_Samples_Mean, name_no_ext);
        
    elapsed_time = toc;
    fprintf('  > GPU Processing completed in %.2f seconds\n', elapsed_time);
    
    % Visualization
    visualize_results(img_double, feature_frobenius, name_no_ext, Vis_Params);
end

fprintf('All files processed successfully!\n');

%% Core Function: GPU Riemannian Frobenius Feature Extraction
function feature_map = extractRiemannianFrobenius_GPU(image_gpu, window_size, reg, ... 
    max_iter, tol, scales, use_multiscale, max_samples, image_name)
    [H, W] = size(image_gpu);
    
    % 1. Compute Gradients on GPU
    [Ix, Iy] = imgradientxy(image_gpu, 'sobel');
    [Ixx, ~] = imgradientxy(Ix, 'sobel');
    [~, Iyy] = imgradientxy(Iy, 'sobel');
    grad_mag = sqrt(Ix.^2 + Iy.^2);
    
    % 2. Compute Covariance Matrix Field
    if use_multiscale
        cov_data_all = computeMultiscaleCovField_GPU(image_gpu, Ix, Iy, Ixx, Iyy, ... 
            grad_mag, scales, reg, image_name);
    else
        cov_data_all = computeCovField_GPU(image_gpu, Ix, Iy, Ixx, Iyy, grad_mag, ... 
            window_size, reg, image_name);
    end
    
    % 3. Transfer Data to CPU for Complex Algebra
    % Covariance data size is 8 by 8 by N pixels
    fprintf('    [%s] Gathering data from GPU to CPU for manifold algebra...\n', image_name);
    cov_field_cpu = gather(cov_data_all); 
    clear cov_data_all; % Free GPU memory
    
    % Reshape for processing
    num_pixels = H * W;
    
    % 4. Compute Global Riemannian Mean
    fprintf('    [%s] Computing Riemannian mean...\n', image_name);
    % Sample indices for mean estimation
    rng(42);
    sample_indices = randperm(num_pixels, min(num_pixels, max_samples));
    sample_matrices = cov_field_cpu(:, :, sample_indices);
    
    % Convert to cell for compatibility with existing mean function
    N_samples = size(sample_matrices, 3);
    cell_samples = cell(1, N_samples);
    for k = 1:N_samples
        cell_samples{k} = sample_matrices(:,:,k);
    end
    global_mean = computeRiemannianMeanWithProgress(cell_samples, max_iter, tol, image_name);
    
    % 5. Extract Features
    fprintf('    [%s] Extracting tangent space features...\n', image_name);    
    feature_vec = zeros(num_pixels, 1);
    
    % Use parfor for massive speedup on pixel-wise eig or logm
    M = global_mean; 
    
    % Parallel Loop
    parfor k = 1:num_pixels
        % Extract single matrix
        C = cov_field_cpu(:, :, k);
        
        % Log map and frobenius norm
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
    feature_map = (feature_map - min(feature_map(:))) / (max(feature_map(:)) - min(feature_map(:)));
end

%% Helper: Vectorized GPU Covariance Computation
function cov_stack = computeCovField_GPU(I, Ix, Iy, Ixx, Iyy, grad, win_size, reg, ~)
    % Input: All images are gpuArrays
    % Output: Covariance stack is 8 by 8 by total pixels double
    [H, W] = size(I);
    num_pixels = H*W;
    
    % Define kernels for convolution
    K_1 = ones(win_size, win_size, 'gpuArray'); 
    N = win_size^2;
    
    % Coordinate filters for local x, y covariance
    half = floor(win_size/2);
    [xx, yy] = meshgrid(-half:half, -half:half);
    K_x = gpuArray(xx / half); % Normalized coordinates
    K_y = gpuArray(yy / half);
    
    % Stack image-based channels    
    Channels = {[], [], I, Ix, Iy, Ixx, Iyy, grad};
    n_ch = 8;    
    cov_stack = zeros(n_ch, n_ch, num_pixels, 'single'); % Use single on GPU to save memory, convert later
    if canUseGPU() && gpuDevice().AvailableMemory > 8e9
        cov_stack = zeros(n_ch, n_ch, num_pixels, 'double');
    end
    
    % Helper for convolution
    do_conv = @(img, k) conv2(img, k, 'same');
    
    % Loop over matrix elements    
    for r = 1:n_ch
        for c = r:n_ch
            % Calculate covariance            
            Term_XY = [];
            Term_X_Mean = [];
            Term_Y_Mean = [];
            
            % --- Analyze Row ---
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
            else % Image vs image
                 
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
    grad, scales, reg, image_name)
    [H, W] = size(I);
    num_pixels = H*W;
    n_ch = 8;
    
    % To keep GPU efficiency, we will calculate all 3 stacks
    fprintf('    [%s] Computing multi-scale convolutions on GPU...\n', image_name);
    
    stack_accum = zeros(n_ch, n_ch, num_pixels, 'double'); % CPU accumulator if GPU RAM low, or GPU
    if canUseGPU(), stack_accum = gpuArray(stack_accum); end
    
    for s = 1:length(scales)
        current_stack = computeCovField_GPU(I, Ix, Iy, Ixx, Iyy, grad, scales(s), reg, image_name);
        stack_accum = stack_accum + current_stack;
    end
    
    % Simplification for speed: Arithmetic mean of covariances
    cov_stack_final = stack_accum / length(scales);
end

%% Helper: Riemannian Mean
function mean_C = computeRiemannianMeanWithProgress(matrices, max_iter, tol, image_name)
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
        
        % This loop is small, simple sequential is fine
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

%% Helper Function: Visualization
function visualize_results(original_img, feature_img, image_name, Vis_Params)
    [H, W] = size(original_img);
    % Coordinate logic
    if contains(image_name, 'RTM', 'IgnoreCase', true)
        axis_x = linspace(0, 1, W); axis_y = linspace(0, 6, H);
        lx = 'Time (s)'; ly = 'Range (m)';
    elseif contains(image_name, 'DTM', 'IgnoreCase', true)
        axis_x = linspace(0, 1, W); axis_y = linspace(-100, 100, H);
        lx = 'Time (s)'; ly = 'Doppler (Hz)';
    elseif contains(image_name, 'RDM', 'IgnoreCase', true)
        axis_x = linspace(-100, 100, W); axis_y = linspace(0, 6, H);
        lx = 'Doppler (Hz)'; ly = 'Range (m)';
    else
        axis_x = 1:W; axis_y = 1:H; lx = 'X'; ly = 'Y';
    end

    figure('Name', sprintf('%s GPU Features', image_name), 'Position', [100, 100, 1200, 500], 'Color', 'w');
    
    subplot(1, 2, 1);
    imagesc(axis_x, axis_y, original_img); axis xy;
    colormap(gca, Vis_Params.Colormap); colorbar;
    title(sprintf('Original %s', image_name), 'FontName', Vis_Params.Font_Name, 'FontSize', Vis_Params.Font_Size_Basis);
    xlabel(lx); ylabel(ly);
    
    subplot(1, 2, 2);
    imagesc(axis_x, axis_y, feature_img); axis xy;
    colormap(gca, Vis_Params.Colormap); colorbar;
    title(sprintf('%s Riemann Feature GPU', image_name), 'FontName', Vis_Params.Font_Name, 'FontSize', Vis_Params.Font_Size_Basis);
    xlabel(lx); ylabel(ly);
end

function b = canUseGPU()
    try
        g = gpuDevice;
        b = true;
    catch
        b = false;
    end
end