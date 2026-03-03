%% Data Reading and Processing: Trace Ratio - Group Sparse (TRGS) with Bottom-K Selection
% Former Author: JoeyBG.
% Improved By: JoeyBG.
% Date: 2025-12-05.
% Affiliate: National University of Defense Technology, Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Introduction:
%   This script achieves the complete workflow of reading raw radar data, preprocessing, and visualization.
%   Selection Logic: 
%       Instead of selecting the most discriminative channels, this version utilizes the 
%           "Trace Ratio - Group Sparse" algorithm to identify and fuse the LEAST important channels (Bottom-K).
%       This is typically used for better noise analysis or algorithm validation.
%       Added Range-Doppler Map (RDM) generation, Bottom-K selection, fusion, and visualization.
% 
% References:
%   None.

%% Initialization of MATLAB Script
clear all;
close all;
clc;
disp('---------- © Author: JoeyBG © ----------');

% Load configurations
try
    load("JoeyBG_Colormap.mat");                                            
    load("Array_Configurations\data_info.mat");                             
catch
    warning('Radar configuration file is missed! Use default instead.');
    if ~exist('data_info','var')
        data_info.f0 = 2.5e9; 
        data_info.B = 1.0e9; 
        data_info.PRT = 1e-3; 
        data_info.fs = 4e6; 
        data_info.antenna_posi = zeros(1,1,1);
    end
end

%% Parameter Definition
% Speed of light
c = 3e8;

% Wall parameters
Thickness_Wall = 0.12;    
Wall_Dieletric = 6;       
Wall_Compensation_Dist = Thickness_Wall * (sqrt(Wall_Dieletric) - 1);       

% Data reading parameters
FilePath = "RW_Datas\P1_Gun.NOV";
Total_Readin_Packages = 100;                                                
Fast_Time_Points = 3940;      
Readin_DS_Ratio = 1;                                                        
Process_DS_Ratio = 5;                                                       
Total_DS_Ratio = Readin_DS_Ratio * Process_DS_Ratio;

% Radar parameters
B = data_info.B;              
T = data_info.PRT;            
fs = data_info.fs;            
PRF_Raw = 1/T;                
PRF_Effective = PRF_Raw / Total_DS_Ratio;                                   

% Calculate the range axis
FFT_Points = Fast_Time_Points;          
Range_Resolution = c / (2 * B);
Range_Axis_Raw = ((0:FFT_Points/2-1) * Range_Resolution) / 2;               

% Wall compensation
Range_Axis_Physical = Range_Axis_Raw - Wall_Compensation_Dist;

% Select ROI
Min_Range_ROI = 0;                                                          
Max_Range_ROI = 6;                                                          
[~, Min_Bin_Idx] = min(abs(Range_Axis_Physical - Min_Range_ROI));
[~, Max_Bin_Idx] = min(abs(Range_Axis_Physical - Max_Range_ROI));
Range_Axis_Show = Range_Axis_Physical(Min_Bin_Idx:Max_Bin_Idx);             
MTI_Step = 1;                                                               

% STFT parameters (Also used for RDM Doppler dimension)
Window_Len_Sec = 0.1;                                                       
Window_Len_Points = floor(Window_Len_Sec * PRF_Effective);                  
Overlap_Ratio = 0.9;                                                        
Overlap_Points = floor(Window_Len_Points * Overlap_Ratio);                  
STFT_FFT_Len = 1024;                                                        
STFT_Win = hamming(Window_Len_Points, "periodic");                          

% Fusion parameters
wname = 'db4';                                                              
wmethod = 'mean';                                                           
level = 2;                                                                  

% TR feature selection parameters
TR_Top_K = 8;                                                               % Number of LEAST important channels to select
TR_Lambda = 0.1;                                                            % Sparse regularization parameter
TR_SubspaceDim = 5;                                                         % Dimension of the latent subspace
TR_Samples = 8192;                                                          % Number of pixels to subsample for graph construction

% Visualization parameters
Font_Name = 'Palatino Linotype';                                            
Font_Size_Basis = 12;                                                       
Font_Size_Axis = 14;                                                        
Font_Size_Title = 16;                                                       
Font_Weight_Basis = 'normal';                                               
Font_Weight_Axis = 'normal';                                                
Font_Weight_Title = 'bold';                                                 

%% Data Reading
fprintf('Read the data file: %s ...\n', FilePath);
[adc_data_raw, ~] = Read_NOV(FilePath, Total_Readin_Packages, 8, 8, Fast_Time_Points, Readin_DS_Ratio);
adc_data = permute(adc_data_raw, [1 3 2]);                                  % [FastTime, Channels, SlowTime]
[~, Total_Channels, Total_Frames_Read] = size(adc_data);
fprintf('Total Frames: %d, Total Channels: %d\n', Total_Frames_Read, Total_Channels);

%% RTM Generation for All Channels
fprintf('Generate RTMs for all channels...\n');

Range_Profile_Complex = fft(adc_data, FFT_Points, 1);
RTM_ROI_Raw = Range_Profile_Complex(Min_Bin_Idx:Max_Bin_Idx, :, :); 

% MTI filtering
RTM_MTI = RTM_ROI_Raw(:, :, MTI_Step+1:end) - RTM_ROI_Raw(:, :, 1:end-MTI_Step);
[nRange, ~, Final_Frames] = size(RTM_MTI);
Time_Axis_RTM = (0 : Final_Frames - MTI_Step - 1) * (T * Total_DS_Ratio); 

% Pre-process RTM Magnitudes
RTM_Mag_Stack = zeros(nRange, Final_Frames, Total_Channels);
for ch = 1:Total_Channels
    temp = abs(squeeze(RTM_MTI(:, ch, :)));
    RTM_Mag_Stack(:,:,ch) = mat2gray(temp); % Normalize to [0, 1]
end

%% RTM Feature Selection and Fusion (Bottom-K)
fprintf('Start RTM feature selection...\n');

% Reshape data for feature selection: [Features (Channels) x Samples (Pixels)]
RTM_Data_Matrix = zeros(Total_Channels, nRange * Final_Frames); 
for ch = 1:Total_Channels
    img = RTM_Mag_Stack(:,:,ch);
    RTM_Data_Matrix(ch, :) = img(:)';
end

% Run TRGS selection
[rtm_channel_scores, rtm_sorted_indices] = TraceRatio_GroupSparse_Selection(...
    RTM_Data_Matrix, TR_Top_K, TR_Lambda, TR_SubspaceDim, TR_Samples);

% Select the LAST K channels (Bottom-K)
selected_indices_RTM = rtm_sorted_indices(end-TR_Top_K+1 : end);
selected_indices_RTM = flip(selected_indices_RTM); 

fprintf('  > RTM Bottom %d Channels Selected: %s\n', TR_Top_K, num2str(selected_indices_RTM));

% Fusion process
Ref_Idx_RTM = selected_indices_RTM(1); 
Reference_RTM = RTM_Mag_Stack(:,:,Ref_Idx_RTM); 

fprintf('  > Fusing Bottom-K RTM channels...\n');
Enhanced_RTM = Reference_RTM;
for i = 2:length(selected_indices_RTM)
    ch_idx = selected_indices_RTM(i);
    Enhanced_RTM = wfusimg(Enhanced_RTM, RTM_Mag_Stack(:,:,ch_idx), wname, level, wmethod, wmethod);
end

%% DTM Generation for All Channels
fprintf('Generate DTMs for all channels...\n');

if Window_Len_Points > length(Time_Axis_RTM)
    Window_Len_Points = length(Time_Axis_RTM) - 2;
    STFT_Win = hamming(Window_Len_Points, "periodic");
    Overlap_Points = floor(Window_Len_Points * Overlap_Ratio);
end

% Determine output size
temp_sig = sum(squeeze(RTM_MTI(:, 1, :)), 1);
[s_temp, f_freq, t_time] = stft(temp_sig, PRF_Effective, ...
    'Window', STFT_Win, 'OverlapLength', Overlap_Points, ...
    'FFTLength', STFT_FFT_Len); 
DTM_Time_Axis = t_time;
[nFreq, nTimeDTM] = size(s_temp);
DTM_Mag_Stack = zeros(nFreq, nTimeDTM, Total_Channels);

for ch = 1:Total_Channels
    Raw_Signal_1D = sum(squeeze(RTM_MTI(:, ch, :)), 1); 
    [S_stft, ~, ~] = stft(Raw_Signal_1D, PRF_Effective, ...
        'Window', STFT_Win, 'OverlapLength', Overlap_Points, ...
        'FFTLength', STFT_FFT_Len);
    DTM_Mag_Stack(:,:,ch) = mat2gray(abs(S_stft)); 
end

%% DTM Feature Selection and Fusion (Bottom-K)
fprintf('Start DTM Feature Selection...\n');

% Reshape data for feature selection
DTM_Data_Matrix = zeros(Total_Channels, nFreq * nTimeDTM);
for ch = 1:Total_Channels
    img = DTM_Mag_Stack(:,:,ch);
    DTM_Data_Matrix(ch, :) = img(:)';
end

% Run TRGS selection
[dtm_channel_scores, dtm_sorted_indices] = TraceRatio_GroupSparse_Selection(...
    DTM_Data_Matrix, TR_Top_K, TR_Lambda, TR_SubspaceDim, TR_Samples);

% Select the LAST K channels
selected_indices_DTM = dtm_sorted_indices(end-TR_Top_K+1 : end);
selected_indices_DTM = flip(selected_indices_DTM); 

fprintf('  > DTM Bottom %d Channels Selected: %s\n', TR_Top_K, num2str(selected_indices_DTM));

% Fusion process
Ref_Idx_DTM = selected_indices_DTM(1); 
Reference_DTM = DTM_Mag_Stack(:,:,Ref_Idx_DTM);

fprintf('  > Fusing Bottom-K DTM channels...\n');
Enhanced_DTM = Reference_DTM;
for i = 2:length(selected_indices_DTM)
    ch_idx = selected_indices_DTM(i);
    Enhanced_DTM = wfusimg(Enhanced_DTM, DTM_Mag_Stack(:,:,ch_idx), wname, level, wmethod, wmethod);
end

%% RDM Generation for All Channels
fprintf('Generate RDMs for all channels...\n');

% Calculate Doppler Axis
Doppler_Axis = linspace(-PRF_Effective/2, PRF_Effective/2, STFT_FFT_Len);

% Initialize Stack [Range, Doppler, Channels]
RDM_Mag_Stack = zeros(nRange, STFT_FFT_Len, Total_Channels);

% Window function for Slow Time dimension to reduce sidelobes
Win_Doppler = hamming(Final_Frames); 

for ch = 1:Total_Channels
    % Extract Range-SlowTime matrix for current channel [Range x Time]
    Current_RTM_Cpx = squeeze(RTM_MTI(:, ch, :));
    
    % Apply windowing along Slow Time
    Current_RTM_Cpx_Win = Current_RTM_Cpx .* Win_Doppler.';
    
    % Perform FFT along Slow Time dimension
    RDM_Complex = fftshift(fft(Current_RTM_Cpx_Win, STFT_FFT_Len, 2), 2);
    
    % Compute Magnitude and Normalize
    RDM_Mag_Stack(:,:,ch) = mat2gray(abs(RDM_Complex));
end

%% RDM Feature Selection and Fusion (Bottom-K)
fprintf('Start RDM Feature Selection...\n');

% Reshape data for feature selection [Channels x Pixels]
RDM_Data_Matrix = zeros(Total_Channels, nRange * STFT_FFT_Len);
for ch = 1:Total_Channels
    img = RDM_Mag_Stack(:,:,ch);
    RDM_Data_Matrix(ch, :) = img(:)';
end

% Run TRGS selection
[rdm_channel_scores, rdm_sorted_indices] = TraceRatio_GroupSparse_Selection(...
    RDM_Data_Matrix, TR_Top_K, TR_Lambda, TR_SubspaceDim, TR_Samples);

% Select the LAST K channels (Bottom-K)
selected_indices_RDM = rdm_sorted_indices(end-TR_Top_K+1 : end);
selected_indices_RDM = flip(selected_indices_RDM); % Absolute worst is first

fprintf('  > RDM Bottom %d Channels Selected: %s\n', TR_Top_K, num2str(selected_indices_RDM));

% Fusion process
Ref_Idx_RDM = selected_indices_RDM(1); % The channel with the lowest score
Reference_RDM = RDM_Mag_Stack(:,:,Ref_Idx_RDM);

fprintf('  > Fusing Bottom-K RDM channels...\n');
Enhanced_RDM = Reference_RDM;
for i = 2:length(selected_indices_RDM)
    ch_idx = selected_indices_RDM(i);
    Enhanced_RDM = wfusimg(Enhanced_RDM, RDM_Mag_Stack(:,:,ch_idx), wname, level, wmethod, wmethod);
end

%% Visualization
fprintf('Start visualizing...\n');
figure('Name', 'TR-GS Bottom-K Multi-Domain Fusion Results', 'Color', 'w', 'Position', [50, 50, 1200, 900]);

% Helper function for normalized log visualization
get_log_display = @(img) log((img - min(img(:))) / (max(img(:)) - min(img(:))) + 1e-6);

% --- Row 1: RTM ---
subplot(3, 2, 1);
imagesc(Time_Axis_RTM, Range_Axis_Show, get_log_display(Reference_RTM));
axis xy; 
if exist('CList_Flip', 'var'), colormap(gca, CList_Flip); else, colormap(gca, jet); end
colorbar; clim([-3, 0]);
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'FontWeight', Font_Weight_Basis, 'LineWidth', 1.5);
title(['Bottom-1 RTM (Ch ', num2str(Ref_Idx_RTM), ')'], 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
xlabel('Time (s)', 'FontSize', Font_Size_Axis); ylabel('Range (m)', 'FontSize', Font_Size_Axis);

subplot(3, 2, 2);
imagesc(Time_Axis_RTM, Range_Axis_Show, get_log_display(Enhanced_RTM));
axis xy; 
if exist('CList_Flip', 'var'), colormap(gca, CList_Flip); else, colormap(gca, jet); end
colorbar; clim([-3, 0]);
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'FontWeight', Font_Weight_Basis, 'LineWidth', 1.5);
title(['Bottom-K RTM (K=', num2str(TR_Top_K), ')'], 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
xlabel('Time (s)', 'FontSize', Font_Size_Axis); ylabel('Range (m)', 'FontSize', Font_Size_Axis);

% --- Row 2: DTM ---
subplot(3, 2, 3);
imagesc(DTM_Time_Axis, f_freq, get_log_display(Reference_DTM));
axis xy;
if exist('CList_Flip', 'var'), colormap(gca, CList_Flip); else, colormap(gca, jet); end
colorbar; clim([-3, 0]);
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'FontWeight', Font_Weight_Basis, 'LineWidth', 1.5);
title(['Bottom-1 DTM (Ch ', num2str(Ref_Idx_DTM), ')'], 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
xlabel('Time (s)', 'FontSize', Font_Size_Axis); ylabel('Doppler (Hz)', 'FontSize', Font_Size_Axis);
ylim([-100, 100]);

subplot(3, 2, 4);
imagesc(DTM_Time_Axis, f_freq, get_log_display(Enhanced_DTM));
axis xy;
if exist('CList_Flip', 'var'), colormap(gca, CList_Flip); else, colormap(gca, jet); end
colorbar; clim([-3, 0]);
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'FontWeight', Font_Weight_Basis, 'LineWidth', 1.5);
title(['Bottom-K DTM (K=', num2str(TR_Top_K), ')'], 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
xlabel('Time (s)', 'FontSize', Font_Size_Axis); ylabel('Doppler (Hz)', 'FontSize', Font_Size_Axis);
ylim([-100, 100]);

% --- Row 3: RDM ---
subplot(3, 2, 5);
imagesc(Doppler_Axis, Range_Axis_Show, get_log_display(Reference_RDM));
axis xy;
if exist('CList_Flip', 'var'), colormap(gca, CList_Flip); else, colormap(gca, jet); end
colorbar; clim([-3, 0]);
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'FontWeight', Font_Weight_Basis, 'LineWidth', 1.5);
title(['Bottom-1 RDM (Ch ', num2str(Ref_Idx_RDM), ')'], 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
xlabel('Doppler (Hz)', 'FontSize', Font_Size_Axis); ylabel('Range (m)', 'FontSize', Font_Size_Axis);
xlim([-100, 100]); ylim([Min_Range_ROI, Max_Range_ROI]);

subplot(3, 2, 6);
imagesc(Doppler_Axis, Range_Axis_Show, get_log_display(Enhanced_RDM));
axis xy;
if exist('CList_Flip', 'var'), colormap(gca, CList_Flip); else, colormap(gca, jet); end
colorbar; clim([-3, 0]);
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'FontWeight', Font_Weight_Basis, 'LineWidth', 1.5);
title(['Bottom-K RDM (K=', num2str(TR_Top_K), ')'], 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
xlabel('Doppler (Hz)', 'FontSize', Font_Size_Axis); ylabel('Range (m)', 'FontSize', Font_Size_Axis);
xlim([-100, 100]); ylim([Min_Range_ROI, Max_Range_ROI]);

fprintf('Visualization completes!\n');

%% Helper Function: TRGS Feature Selection
function [scores, sorted_indices] = TraceRatio_GroupSparse_Selection(X_raw, ~, lambda, m, n_samples)
    % Inputs:
    %   X_raw: Data matrix [d_features x N_samples] 
    %   lambda: Regularization parameter for Group Sparsity
    %   m: Dimension of the subspace
    %   n_samples: Number of samples to use for graph construction
    %
    % Outputs:
    %   scores: Weight (L2 norm) for each feature
    %   sorted_indices: Indices of features sorted by importance

    [d, N_total] = size(X_raw);
    
    % Data subsampling and normalization
    if N_total > n_samples
        rand_idx = randperm(N_total, n_samples);
        X = X_raw(:, rand_idx);
    else
        X = X_raw;
    end 
    [~, n] = size(X);
    
    % Center the data
    X = X - mean(X, 2);
    
    % Construct graphs
    k = 5; 
    
    % Calculate pairwise Euclidean distances between sample
    dist_matrix = pdist2(X', X').^2; % Transpose X to [n x d] for pdist2
    
    % Build adjacency matrix W_graph
    W_graph = zeros(n, n);
    for i = 1:n
        [~, idx] = sort(dist_matrix(i, :), 'ascend');
        % Connect k neighbors
        nbs = idx(2:k+1);
        % Heat kernel weighting
        sigma = mean(dist_matrix(i, nbs)); % Adaptive sigma
        if sigma == 0, sigma = 1e-5; end
        W_graph(i, nbs) = exp(-dist_matrix(i, nbs) / (2*sigma^2));
    end
    W_graph = (W_graph + W_graph') / 2; % Symmetrize
    D_graph = diag(sum(W_graph, 2)); % Degree matrix
    L = D_graph - W_graph; % Laplacian matrix
    
    % Calculate scatter matrices
    Sw = X * L * X'; % Within-class scatter
    Sw = Sw + 1e-4 * eye(d); % Regularize Sw for stability
    
    % Total scatter
    St = X * X';
    
    % Between-class scatter
    Sb = St - Sw;
    
    % Iterative TR algorithm with L2,1 Norm
    [U_pca, ~, ~] = svd(X, 'econ');
    W = U_pca(:, 1:m);    
    max_iter = 15;
    
    for iter = 1:max_iter
        % Update diagonal matrix D
        d_diag = zeros(d, 1);
        for i = 1:d
            wi_norm = norm(W(i, :), 2);
            d_diag(i) = 1 / (2 * wi_norm + 1e-6); 
        end
        D_sparse = diag(d_diag);
        
        % Calculate current TR value
        num = trace(W' * Sb * W);
        den = trace(W' * (Sw + lambda * D_sparse) * W);
        if den == 0, den = 1e-6; end
        eta = num / den;
        
        % Solve eigenvalue problem
        P = Sb - eta * (Sw + lambda * D_sparse);
        
        % Force symmetry for numerical stability
        P = (P + P') / 2;        
        [V, E_val] = eig(P);
        [~, idx] = sort(diag(E_val), 'descend');
        W = V(:, idx(1:m));
        
        % Orthogonalize W
        [W, ~] = qr(W, 0);
    end
    
    % Feature scoring
    scores = zeros(d, 1);
    for i = 1:d
        scores(i) = norm(W(i, :), 2);
    end
    
    % Sort Descending
    [~, sorted_indices] = sort(scores, 'descend');
end