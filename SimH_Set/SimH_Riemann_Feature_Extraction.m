%% Riemannian Manifold Feature Extraction Script: Riemann-Based RTM/DTM/RDM Processor
% Original Author: JoeyBG. 
% Improved By: JoeyBG. 
% Date: 2025-12-05. 
% Affiliate: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Introduction:
%   This script extracts fine-grained features from RTM, DTM, and RDM images using
%       Riemannian geometry on the Symmetric Positive Definite (SPD) manifold.
%   Methodology:
%       Local covariance descriptors are computed at each pixel, forming SPD matrices.
%       The Log-Euclidean framework maps these matrices to tangent space via Log map.
%       Frobenius norm of tangent vectors yields the final feature image.
%   Output:
%       Visualization of original images and Riemannian feature images.
%
% Files to process:
%   RTM.png, DTM.png, RDM.png located in 'Example_Images' folder. 

%% Initialization of MATLAB Script
clear all;
close all;
clc;
disp('---------- © Author: JoeyBG © ----------');

% Load configurations
try
    load("JoeyBG_Colormap.mat");                                            % My favorite colormap
catch
    warning('JoeyBG_Colormap.mat is missed! Use jet instead.');
end

%% Parameter Definitions
% File and directory settings
Data_Folder = "Example_Images";                                             % Input folder
File_List = ["RTM.png", "DTM.png", "RDM.png"];                              % Files to process (Added RDM)

% Riemannian feature extraction parameters
Window_Size = 5;                                                            % Local window size for covariance computation
Regularization = 1e-6;                                                      % Regularization for SPD guarantee
Max_Iter_Mean = 15;                                                         % Maximum iterations for Riemannian mean
Tol_Mean = 1e-6;                                                            % Convergence tolerance for Riemannian mean
Scales = [3, 5, 7];                                                         % Multi-scale window sizes
Use_Multiscale = true;                                                      % Enable multi-scale processing
Max_Samples_Mean = 1024;                                                    % Maximum samples for Riemannian mean estimation

% Visualization parameters
Vis_Params.Font_Name = 'Palatino Linotype';                                 % Font name used for plotting
Vis_Params.Font_Size_Basis = 15;                                            % Base font size
Vis_Params.Font_Size_Axis = 16;                                             % Font size for axis labels
Vis_Params.Font_Size_Title = 18;                                            % Font size for the title
Vis_Params.Font_Weight_Basis = 'normal';                                    % Base font weight
Vis_Params.Font_Weight_Axis = 'normal';                                     % Font weight for axis labels
Vis_Params.Font_Weight_Title = 'bold';                                      % Font weight for the title
if exist('CList_Flip', 'var')
    Vis_Params.Colormap = CList_Flip;
else
    Vis_Params.Colormap = jet;
end

%% Main Processing Loop
fprintf('Starting Riemannian Feature Extraction...\n');
total_files = length(File_List);

for f_idx = 1:total_files
    current_file_name = File_List(f_idx);
    full_path = fullfile(Data_Folder, current_file_name);
    
    [~, name_no_ext, ~] = fileparts(current_file_name);
    
    fprintf('Processing File (%d/%d): %s\n', f_idx, total_files, current_file_name);
    
    % Read and preprocess image
    try
        img_original = imread(full_path);
    catch
        warning('File %s not found. Skipping.', full_path);
        continue;
    end
    
    % Convert to grayscale if necessary
    if size(img_original, 3) == 3
        img_gray = rgb2gray(img_original);
    else
        img_gray = img_original;
    end
    
    % Convert to double precision [0, 1]
    img_double = im2double(img_gray);
    img_double = (img_double - min(min(img_double)))... 
        / (max(max(img_double)) - min(min(img_double)));
    [H, W] = size(img_double);
    fprintf('  > Image size: %d x %d\n', H, W);
    
    % Extract Riemannian features
    tic;
    feature_frobenius = extractRiemannianFrobenius(img_double, Window_Size, ... 
        Regularization, Max_Iter_Mean, Tol_Mean, Scales, Use_Multiscale, ... 
        Max_Samples_Mean, name_no_ext);
    elapsed_time = toc;
    fprintf('  > Feature extraction completed in %.2f seconds\n', elapsed_time);
    
    % Visualization with updated parameters
    visualize_results(img_double, feature_frobenius, name_no_ext, Vis_Params);
end

fprintf('All files processed successfully!\n');

%% Core Function: Riemannian Frobenius Feature Extraction
function feature_map = extractRiemannianFrobenius(image, window_size, reg, ... 
    max_iter, tol, scales, use_multiscale, max_samples, image_name)
    % extractRiemannianFrobenius - Extract Frobenius norm features via Riemannian geometry    
    [H, W] = size(image);
    
    % Compute image gradients
    [Ix, Iy] = imgradientxy(image, 'sobel');
    [Ixx, ~] = imgradientxy(Ix, 'sobel');
    [~, Iyy] = imgradientxy(Iy, 'sobel');
    grad_mag = sqrt(Ix.^2 + Iy.^2);
    
    % Compute covariance matrix field
    if use_multiscale
        cov_field = computeMultiscaleCovField(image, Ix, Iy, Ixx, Iyy, ... 
            grad_mag, scales, reg, image_name);
    else
        cov_field = computeCovField(image, Ix, Iy, Ixx, Iyy, grad_mag, ... 
            window_size, reg, image_name);
    end
    
    % Compute global Riemannian mean
    fprintf('    [%s] Computing Riemannian mean...\n', image_name);
    global_mean = computeRiemannianMean(cov_field, max_iter, tol, max_samples, image_name);
    
    % Extract Frobenius norm features via Log map
    fprintf('    [%s] Extracting tangent space features...\n', image_name);
    feature_map = zeros(H, W);
    total_pixels = H * W;
    
    % Create progress bar
    hWait = waitbar(0, sprintf('[%s] Extracting features: 0%%', image_name), ...
        'Name', sprintf('%s - Feature Extraction', image_name));
    update_interval = max(1, floor(total_pixels / 100));
    
    for i = 1:H
        for j = 1:W
            C = cov_field{i, j};
            
            % Log map to tangent space
            S = logMap(global_mean, C);
            
            % Frobenius norm
            feature_map(i, j) = norm(S, 'fro');
            
            % Update progress bar
            current_pixel = (i - 1) * W + j;
            if mod(current_pixel, update_interval) == 0 || current_pixel == total_pixels
                progress = current_pixel / total_pixels;
                waitbar(progress, hWait, sprintf('[%s] Extracting features: %.1f%%', ... 
                    image_name, progress * 100));
            end
        end
    end
    
    close(hWait);
    
    % Normalize to [0, 1]
    feature_map = normalizeImage(feature_map);
end

%% Helper Function: Single-Scale Covariance Field Computation
function cov_field = computeCovField(image, Ix, Iy, Ixx, Iyy, grad_mag, ...
    window_size, reg, image_name)
    
    [H, W] = size(image);
    half_win = floor(window_size / 2);
    cov_field = cell(H, W);
    
    image_pad = padarray(image, [half_win, half_win], 'symmetric');
    Ix_pad = padarray(Ix, [half_win, half_win], 'symmetric');
    Iy_pad = padarray(Iy, [half_win, half_win], 'symmetric');
    Ixx_pad = padarray(Ixx, [half_win, half_win], 'symmetric');
    Iyy_pad = padarray(Iyy, [half_win, half_win], 'symmetric');
    grad_mag_pad = padarray(grad_mag, [half_win, half_win], 'symmetric');
    
    total_pixels = H * W;
    hWait = waitbar(0, sprintf('[%s] Computing covariance field: 0%%', image_name), ... 
        'Name', sprintf('%s - Covariance Computation', image_name));
    update_interval = max(1, floor(total_pixels / 100));
    
    for i = 1:H
        for j = 1:W
            row_range = i:(i + 2*half_win);
            col_range = j:(j + 2*half_win);
            
            patch_I = image_pad(row_range, col_range);
            patch_Ix = Ix_pad(row_range, col_range);
            patch_Iy = Iy_pad(row_range, col_range);
            patch_Ixx = Ixx_pad(row_range, col_range);
            patch_Iyy = Iyy_pad(row_range, col_range);
            patch_grad = grad_mag_pad(row_range, col_range);
            
            [xx, yy] = meshgrid(1:window_size, 1:window_size);
            xx = (xx(:) - half_win - 1) / half_win;
            yy = (yy(:) - half_win - 1) / half_win;
            
            F = [xx, yy, patch_I(:), patch_Ix(:), patch_Iy(:), ... 
                 patch_Ixx(:), patch_Iyy(:), patch_grad(:)];
            
            F_centered = F - mean(F, 1);
            C = (F_centered' * F_centered) / (size(F, 1) - 1);
            C = C + reg * eye(size(C));
            cov_field{i, j} = C;
            
            current_pixel = (i - 1) * W + j;
            if mod(current_pixel, update_interval) == 0 || current_pixel == total_pixels
                progress = current_pixel / total_pixels;
                waitbar(progress, hWait, sprintf('[%s] Computing covariance field: %.1f%%', ...
                    image_name, progress * 100));
            end
        end
    end
    close(hWait);
end

%% Helper Function: Multi-Scale Covariance Field Computation
function cov_field = computeMultiscaleCovField(image, Ix, Iy, Ixx, Iyy, ... 
    grad_mag, scales, reg, image_name)

    [H, W] = size(image);
    num_scales = length(scales);
    
    fprintf('    [%s] Computing multi-scale covariance fields...\n', image_name);
    cov_fields_scales = cell(1, num_scales);
    for s = 1:num_scales
        fprintf('      Scale %d/%d (window=%d)\n', s, num_scales, scales(s));
        cov_fields_scales{s} = computeCovField(image, Ix, Iy, Ixx, Iyy, ...
            grad_mag, scales(s), reg, sprintf('%s-Scale%d', image_name, s));
    end
    
    fprintf('    [%s] Fusing multi-scale covariances...\n', image_name);
    cov_field = cell(H, W);
    total_pixels = H * W;
    hWait = waitbar(0, sprintf('[%s] Multi-scale fusion: 0%%', image_name), ... 
        'Name', sprintf('%s - Multi-scale Fusion', image_name));
    update_interval = max(1, floor(total_pixels / 100));
    
    for i = 1:H
        for j = 1:W
            matrices = cell(1, num_scales);
            for s = 1:num_scales
                matrices{s} = cov_fields_scales{s}{i, j};
            end
            cov_field{i, j} = computeRiemannianMeanSmall(matrices, 10, 1e-5);
            
            current_pixel = (i - 1) * W + j;
            if mod(current_pixel, update_interval) == 0 || current_pixel == total_pixels
                progress = current_pixel / total_pixels;
                waitbar(progress, hWait, sprintf('[%s] Multi-scale fusion: %.1f%%', ... 
                    image_name, progress * 100));
            end
        end
    end
    close(hWait);
end

%% Helper Function: Riemannian Mean Computation
function mean_C = computeRiemannianMean(cov_field, max_iter, tol, max_samples, image_name)
    [H, W] = size(cov_field);
    N = H * W;
    if N > max_samples
        indices = randperm(N, max_samples);
        fprintf('      Using random sampling: %d/%d matrices\n', max_samples, N);
    else
        indices = 1:N;
    end
    all_matrices = cell(1, length(indices));
    hWait = waitbar(0, sprintf('[%s] Preparing Riemannian mean: 0%%', image_name), ... 
        'Name', sprintf('%s - Riemannian Mean', image_name));
    for k = 1:length(indices)
        idx = indices(k);
        [i, j] = ind2sub([H, W], idx);
        all_matrices{k} = cov_field{i, j};
        if mod(k, 100) == 0 || k == length(indices)
            waitbar(k / length(indices) * 0.3, hWait, ... 
                sprintf('[%s] Preparing Riemannian mean: %.1f%%', image_name, k / length(indices) * 30));
        end
    end
    waitbar(0.3, hWait, sprintf('[%s] Iterating Riemannian mean... ', image_name));
    mean_C = computeRiemannianMeanWithProgress(all_matrices, max_iter, tol, hWait, image_name);
    close(hWait);
end

%% Helper Function: Riemannian Mean with Progress Bar
function mean_C = computeRiemannianMeanWithProgress(matrices, max_iter, tol, hWait, image_name)   
    N = length(matrices);
    d = size(matrices{1}, 1);
    mean_C = zeros(d);
    for k = 1:N
        mean_C = mean_C + matrices{k};
    end
    mean_C = mean_C / N;
    mean_C = (mean_C + mean_C') / 2;
    [V, D] = eig(mean_C);
    D = diag(max(diag(D), 1e-6));
    mean_C = V * D * V';
    for iter = 1:max_iter
        progress = 0.3 + 0.7 * (iter / max_iter);
        waitbar(progress, hWait, sprintf('[%s] Riemannian mean iteration: %d/%d', ...
            image_name, iter, max_iter));
        [V, D] = eig(mean_C);
        D = diag(max(diag(D), 1e-10));
        sqrt_C = V * diag(sqrt(diag(D))) * V';
        inv_sqrt_C = V * diag(1./sqrt(diag(D))) * V';
        tangent_mean = zeros(d);
        for k = 1:N
            M = inv_sqrt_C * matrices{k} * inv_sqrt_C;
            M = (M + M') / 2;
            [Vm, Dm] = eig(M);
            Dm = diag(max(diag(Dm), 1e-10));
            log_M = Vm * diag(log(diag(Dm))) * Vm';
            tangent_mean = tangent_mean + log_M;
        end
        tangent_mean = tangent_mean / N;
        tangent_norm = norm(tangent_mean, 'fro');
        if tangent_norm < tol
            fprintf('      Riemannian mean converged at iteration %d (residual: %.2e)\n', ... 
                iter, tangent_norm);
            break;
        end
        [Vt, Dt] = eig(tangent_mean);
        exp_tangent = Vt * diag(exp(diag(Dt))) * Vt';
        mean_C = sqrt_C * exp_tangent * sqrt_C;
        mean_C = (mean_C + mean_C') / 2;
    end
end

%% Helper Function: Small-Scale Riemannian Mean
function mean_C = computeRiemannianMeanSmall(matrices, max_iter, tol)    
    N = length(matrices);
    d = size(matrices{1}, 1);
    mean_C = zeros(d);
    for k = 1:N
        mean_C = mean_C + matrices{k};
    end
    mean_C = mean_C / N;
    mean_C = (mean_C + mean_C') / 2;
    [V, D] = eig(mean_C);
    D = diag(max(diag(D), 1e-6));
    mean_C = V * D * V';
    for iter = 1:max_iter
        [V, D] = eig(mean_C);
        D = diag(max(diag(D), 1e-10));
        sqrt_C = V * diag(sqrt(diag(D))) * V';
        inv_sqrt_C = V * diag(1./sqrt(diag(D))) * V';
        tangent_mean = zeros(d);
        for k = 1:N
            M = inv_sqrt_C * matrices{k} * inv_sqrt_C;
            M = (M + M') / 2;
            [Vm, Dm] = eig(M);
            Dm = diag(max(diag(Dm), 1e-10));
            log_M = Vm * diag(log(diag(Dm))) * Vm';
            tangent_mean = tangent_mean + log_M;
        end
        tangent_mean = tangent_mean / N;
        if norm(tangent_mean, 'fro') < tol
            break;
        end
        [Vt, Dt] = eig(tangent_mean);
        exp_tangent = Vt * diag(exp(diag(Dt))) * Vt';
        mean_C = sqrt_C * exp_tangent * sqrt_C;
        mean_C = (mean_C + mean_C') / 2;
    end
end

%% Helper Function: Logarithmic Map on SPD Manifold
function S = logMap(P, Q)    
    [V, D] = eig(P);
    D = diag(max(diag(D), 1e-10));
    sqrt_P = V * diag(sqrt(diag(D))) * V';
    inv_sqrt_P = V * diag(1./sqrt(diag(D))) * V';
    M = inv_sqrt_P * Q * inv_sqrt_P;
    M = (M + M') / 2;
    [Vm, Dm] = eig(M);
    Dm = diag(max(diag(Dm), 1e-10));
    log_M = Vm * diag(log(diag(Dm))) * Vm';
    S = sqrt_P * log_M * sqrt_P;
    S = (S + S') / 2;
end

%% Helper Function: Image Normalization
function img_norm = normalizeImage(img)    
    min_val = min(img(:));
    max_val = max(img(:));
    if max_val - min_val < 1e-10
        img_norm = zeros(size(img));
    else
        img_norm = (img - min_val) / (max_val - min_val);
    end
end

%% Helper Function: Visualization
function visualize_results(original_img, feature_img, image_name, Vis_Params)
    % visualize_results - Visualize comparison with physical axes and styling
    [H, W] = size(original_img);
    
    % Default assumptions for generic files
    axis_x = 1:W;
    axis_y = 1:H;
    label_x = 'X (px)';
    label_y = 'Y (px)';
    
    if contains(image_name, 'RTM', 'IgnoreCase', true)
        % RTM: Time (0-1s) vs Range (0-6m)
        axis_x = linspace(0, 1, W);
        axis_y = linspace(0, 6, H);
        label_x = 'Time (s)';
        label_y = 'Range (m)';
    elseif contains(image_name, 'DTM', 'IgnoreCase', true)
        % DTM: Time (0-1s) vs Doppler (-100-100Hz)
        axis_x = linspace(0, 1, W);
        axis_y = linspace(-100, 100, H);
        label_x = 'Time (s)';
        label_y = 'Doppler (Hz)';
    elseif contains(image_name, 'RDM', 'IgnoreCase', true)
        % RDM: Doppler (-100-100Hz) vs Range (0-6m)
        axis_x = linspace(-100, 100, W);
        axis_y = linspace(0, 6, H);
        label_x = 'Doppler (Hz)';
        label_y = 'Range (m)';
    end

    figure('Name', sprintf('%s Feature Extraction', image_name), ... 
        'Position', [100, 100, 1200, 500], 'Color', 'w');
    
    % Original Image
    subplot(1, 2, 1);
    imagesc(axis_x, axis_y, original_img);
    axis xy; % Ensure Y-axis increases upwards
    colormap(gca, Vis_Params.Colormap);
    colorbar;
    
    % Apply Visualization Parameters
    set(gca, 'FontName', Vis_Params.Font_Name, 'FontSize', Vis_Params.Font_Size_Basis, ...
        'FontWeight', Vis_Params.Font_Weight_Basis, 'LineWidth', 1.5);
    title(sprintf('Original %s', image_name), ...
        'FontName', Vis_Params.Font_Name, 'FontSize', Vis_Params.Font_Size_Title, ...
        'FontWeight', Vis_Params.Font_Weight_Title);
    xlabel(label_x, ...
        'FontName', Vis_Params.Font_Name, 'FontSize', Vis_Params.Font_Size_Axis, ...
        'FontWeight', Vis_Params.Font_Weight_Axis);
    ylabel(label_y, ...
        'FontName', Vis_Params.Font_Name, 'FontSize', Vis_Params.Font_Size_Axis, ...
        'FontWeight', Vis_Params.Font_Weight_Axis);
        
    % Enforce limits
    xlim([min(axis_x), max(axis_x)]);
    ylim([min(axis_y), max(axis_y)]);
    
    % Riemannian Feature Image
    subplot(1, 2, 2);
    imagesc(axis_x, axis_y, feature_img);
    axis xy;
    colormap(gca, Vis_Params.Colormap);
    colorbar;
    
    % Apply Visualization Parameters
    set(gca, 'FontName', Vis_Params.Font_Name, 'FontSize', Vis_Params.Font_Size_Basis, ...
        'FontWeight', Vis_Params.Font_Weight_Basis, 'LineWidth', 1.5);
    title(sprintf('%s Riemannian Feature', image_name), ... 
        'FontName', Vis_Params.Font_Name, 'FontSize', Vis_Params.Font_Size_Title, ...
        'FontWeight', Vis_Params.Font_Weight_Title);
    xlabel(label_x, ...
        'FontName', Vis_Params.Font_Name, 'FontSize', Vis_Params.Font_Size_Axis, ...
        'FontWeight', Vis_Params.Font_Weight_Axis);
    ylabel(label_y, ...
        'FontName', Vis_Params.Font_Name, 'FontSize', Vis_Params.Font_Size_Axis, ...
        'FontWeight', Vis_Params.Font_Weight_Axis);
    
    % Enforce limits
    xlim([min(axis_x), max(axis_x)]);
    ylim([min(axis_y), max(axis_y)]);
end