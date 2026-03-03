%% Batch Processing Script: TRGS Bottom-K Fusion Dataset Generator
% Original Author: JoeyBG.
% Modified By: JoeyBG.
% Date: 2025-12-05.
% Affiliate: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Introduction:
%   This script processes raw Novasky MIMO Radar data to generate a dataset of RTM, DTM, and RDM images.
%   Methodology: 
%       Instead of PSNR/Entropy, this version uses "Trace Ratio - Group Sparse"
%           to identify the LEAST important channels (Bottom-K) and fuses them.
%   Output:
%       1024x1024 PNG images with Jet colormap, saved in:
%       'RW_RTM_Set_TRGS', 'RW_DTM_Set_TRGS', and 'RW_RDM_Set_TRGS'.
%
% Files to process:
%   P1_Nogun.NOV, P1_Gun.NOV, P2_Nogun.NOV, P2_Gun.NOV,
%   P3_Nogun.NOV, P3_Gun.NOV, P4_Nogun.NOV, P4_Gun.NOV.

%% Initialization of MATLAB Script
clear all; 
close all; 
clc;
disp('---------- © Author: JoeyBG © ----------');

%% Parameter Definitions
% File and directory settings
Data_Folder = "RW_Datas";
Output_Root_RTM = "RW_RTM_Set_TRGS";                                        
Output_Root_DTM = "RW_DTM_Set_TRGS";                                        
Output_Root_RDM = "RW_RDM_Set_TRGS";                                       

File_List = [
    "P1_Nogun.NOV", "P1_Gun.NOV", ...
    "P2_Nogun.NOV", "P2_Gun.NOV", ...
    "P3_Nogun.NOV", "P3_Gun.NOV", ...
    "P4_Nogun.NOV", "P4_Gun.NOV"
];

% Reading parameters
Window_Packets = 100;                                                       % Read Window_Packets packages per group
Step_Packets = 25;                                                          % Step Step_Packets packages
Fast_Time_Points = 3940;                                                    % adc_num
Rx_Num = 8;
Tx_Num = 8;
Header_Len = 60;
Bytes_Per_Int16 = 2;

% Calculate block sizes for fseek
DS_Read = 1; 
Points_Per_Block = Rx_Num * (Fast_Time_Points + Header_Len) * Tx_Num * DS_Read;
Bytes_Per_Block = Points_Per_Block * Bytes_Per_Int16;
Bytes_Per_Packet_Arg = Bytes_Per_Block * 2;                                 % Because solving_num = packet*2

% Radar physics and processing parameters
c = 3e8;
try
    load("Array_Configurations\data_info.mat");
catch
    data_info.B = 1.0e9; 
    data_info.PRT = 1e-3; 
    data_info.fs = 4e6; 
end

B = data_info.B;
T = data_info.PRT;
Process_DS_Ratio = 5;
Total_DS_Ratio = DS_Read * Process_DS_Ratio;
PRF_Effective = (1/T) / Total_DS_Ratio;

% Wall compensation
Thickness_Wall = 0.12;    
Wall_Dieletric = 6;       
Wall_Compensation_Dist = Thickness_Wall * (sqrt(Wall_Dieletric) - 1);

% Range axis and ROI
FFT_Points = Fast_Time_Points;
Range_Resolution = c / (2 * B);
Range_Axis_Raw = ((0:FFT_Points/2-1) * Range_Resolution) / 2;
Range_Axis_Physical = Range_Axis_Raw - Wall_Compensation_Dist;

Min_Range_ROI = 0;
Max_Range_ROI = 6;
[~, Min_Bin_Idx] = min(abs(Range_Axis_Physical - Min_Range_ROI));
[~, Max_Bin_Idx] = min(abs(Range_Axis_Physical - Max_Range_ROI));

% MTI
MTI_Step = 1;

% STFT / Doppler parameters
Window_Len_Sec = 0.1;
Window_Len_Points = floor(Window_Len_Sec * PRF_Effective);
Overlap_Ratio = 0.9;
Overlap_Points = floor(Window_Len_Points * Overlap_Ratio);
STFT_FFT_Len = 1024;
STFT_Win = hamming(Window_Len_Points, "periodic");

% Fusion parameters (Wavelet)
wname = 'db4';
level = 2;
wmethod = 'mean';

% TRGS Feature Selection Parameters
TR_Top_K = 8;                                                               % Number of LEAST important channels to select
TR_Lambda = 0.1;                                                            % Sparse regularization parameter
TR_SubspaceDim = 5;                                                         % Dimension of the latent subspace
TR_Samples = 8192;                                                          % Number of pixels to subsample for graph construction

% Image output
Img_Size = [1024, 1024];

%% Main Processing Loop
fprintf('Starting Data Processing (TRGS Bottom-K Mode)...\n');

for f_idx = 1:length(File_List)
    current_file_name = File_List(f_idx);
    full_path = fullfile(Data_Folder, current_file_name);
    
    [~, name_no_ext, ~] = fileparts(current_file_name);
    
    % Create output sub-directories
    save_dir_rtm = fullfile(Output_Root_RTM, name_no_ext);
    save_dir_dtm = fullfile(Output_Root_DTM, name_no_ext);
    save_dir_rdm = fullfile(Output_Root_RDM, name_no_ext); % [NEW]
    
    if ~exist(save_dir_rtm, 'dir'), mkdir(save_dir_rtm); end
    if ~exist(save_dir_dtm, 'dir'), mkdir(save_dir_dtm); end
    if ~exist(save_dir_rdm, 'dir'), mkdir(save_dir_rdm); end % [NEW]
    
    fprintf('Processing File (%d/%d): %s\n', f_idx, length(File_List), current_file_name);
    
    fid = fopen(full_path, 'r');
    if fid == -1
        warning('Cannot open file: %s', full_path);
        continue;
    end
    
    % Check file size
    fseek(fid, 0, 'eof');
    file_size_bytes = ftell(fid);
    fseek(fid, 0, 'bof');
    
    total_packets_in_file = floor(file_size_bytes / Bytes_Per_Packet_Arg);
    
    % Sliding window loop
    group_count = 0;
    
    for start_pkt = 0 : Step_Packets : (total_packets_in_file - Window_Packets)
        group_count = group_count + 1;
        
        % Read data chunk
        offset_bytes = start_pkt * Bytes_Per_Packet_Arg;
        fseek(fid, offset_bytes, 'bof');
        
        solving_num = Window_Packets * 2; 
        
        % Pre-allocate
        adc_data_chunk_native = zeros(Fast_Time_Points, solving_num, Tx_Num*Rx_Num);
        
        read_success = true;
        for ii = 1:solving_num
            frame_data = fread(fid, Points_Per_Block, 'int16');
            if length(frame_data) < Points_Per_Block
                read_success = false; break; 
            end
            
            % Unpack
            frame_data_r = reshape(frame_data, Rx_Num, (Fast_Time_Points+60), Tx_Num, DS_Read);
            A_permuted = permute(frame_data_r(:, 1:Fast_Time_Points, :, :), [1 3 2, 4]); 
            A_permuted = permute(A_permuted, [3, 1, 2, 4]); 
            adc_data_r = reshape(A_permuted, [Fast_Time_Points, Rx_Num*Tx_Num, DS_Read]);
            
            adc_data_chunk_native(:, ii, :) = reshape(adc_data_r(:, :, 1), [Fast_Time_Points, 1, Rx_Num*Tx_Num]);
        end
        
        if ~read_success
            break; 
        end
        
        adc_data = permute(adc_data_chunk_native, [1 3 2]); 
        [~, Total_Channels, Total_Frames] = size(adc_data);
        
        % -----------------------------------------------------------------
        % 1. Generate & Process RTM
        % -----------------------------------------------------------------
        Range_Profile_Complex = fft(adc_data, FFT_Points, 1);
        RTM_ROI_Raw = Range_Profile_Complex(Min_Bin_Idx:Max_Bin_Idx, :, :);
        RTM_MTI = RTM_ROI_Raw(:, :, MTI_Step+1:end) - RTM_ROI_Raw(:, :, 1:end-MTI_Step);
        [nRange, ~, Final_Frames] = size(RTM_MTI);
        
        RTM_Mag_Stack = zeros(nRange, Final_Frames, Total_Channels);
        for ch = 1:Total_Channels
            temp = abs(squeeze(RTM_MTI(:, ch, :)));
            RTM_Mag_Stack(:,:,ch) = mat2gray(temp);
        end
        
        % TRGS Bottom-K for RTM
        RTM_Data_Matrix = zeros(Total_Channels, nRange * Final_Frames);
        for ch = 1:Total_Channels
            img = RTM_Mag_Stack(:,:,ch);
            RTM_Data_Matrix(ch, :) = img(:)';
        end
        
        [~, rtm_sorted_indices] = TraceRatio_GroupSparse_Selection(...
            RTM_Data_Matrix, TR_Top_K, TR_Lambda, TR_SubspaceDim, TR_Samples);
            
        selected_indices_RTM = flip(rtm_sorted_indices(end-TR_Top_K+1 : end)); 
        
        Reference_RTM = RTM_Mag_Stack(:,:,selected_indices_RTM(1));
        Enhanced_RTM = Reference_RTM;
        for i = 2:length(selected_indices_RTM)
            ch_idx = selected_indices_RTM(i);
            Enhanced_RTM = wfusimg(Enhanced_RTM, RTM_Mag_Stack(:,:,ch_idx), wname, level, wmethod, wmethod);
        end

        % -----------------------------------------------------------------
        % 2. Generate & Process DTM
        % -----------------------------------------------------------------
        if Window_Len_Points > Final_Frames
             curr_Win_Len = Final_Frames - 2;
             curr_Win = hamming(curr_Win_Len, "periodic");
             curr_Overlap = floor(curr_Win_Len * Overlap_Ratio);
        else
             curr_Win_Len = Window_Len_Points;
             curr_Win = STFT_Win;
             curr_Overlap = Overlap_Points;
        end
        
        temp_sig = sum(squeeze(RTM_MTI(:, 1, :)), 1);
        [~, f_freq, t_time] = stft(temp_sig, PRF_Effective, ...
            'Window', curr_Win, 'OverlapLength', curr_Overlap, 'FFTLength', STFT_FFT_Len);
        
        DTM_Mag_Stack = zeros(length(f_freq), length(t_time), Total_Channels);
        for ch = 1:Total_Channels
            Raw_Signal_1D = sum(squeeze(RTM_MTI(:, ch, :)), 1);
            [S_stft, ~, ~] = stft(Raw_Signal_1D, PRF_Effective, ...
                'Window', curr_Win, 'OverlapLength', curr_Overlap, 'FFTLength', STFT_FFT_Len);
            DTM_Mag_Stack(:,:,ch) = mat2gray(abs(S_stft));
        end
        
        % TRGS Bottom-K for DTM
        [nFreqActual, nTimeActual, ~] = size(DTM_Mag_Stack);
        DTM_Data_Matrix = zeros(Total_Channels, nFreqActual * nTimeActual);
        for ch = 1:Total_Channels
            img = DTM_Mag_Stack(:,:,ch);
            DTM_Data_Matrix(ch, :) = img(:)';
        end
        
        [~, dtm_sorted_indices] = TraceRatio_GroupSparse_Selection(...
            DTM_Data_Matrix, TR_Top_K, TR_Lambda, TR_SubspaceDim, TR_Samples);
            
        selected_indices_DTM = flip(dtm_sorted_indices(end-TR_Top_K+1 : end));
        
        Reference_DTM = DTM_Mag_Stack(:,:,selected_indices_DTM(1));
        Enhanced_DTM = Reference_DTM;
        for i = 2:length(selected_indices_DTM)
            ch_idx = selected_indices_DTM(i);
            Enhanced_DTM = wfusimg(Enhanced_DTM, DTM_Mag_Stack(:,:,ch_idx), wname, level, wmethod, wmethod);
        end

        % -----------------------------------------------------------------
        % 3. Generate & Process RDM
        % -----------------------------------------------------------------
        RDM_Mag_Stack = zeros(nRange, STFT_FFT_Len, Total_Channels);
        Win_Doppler = hamming(Final_Frames);

        for ch = 1:Total_Channels
            Current_RTM_Cpx = squeeze(RTM_MTI(:, ch, :)); 
            Current_RTM_Cpx_Win = Current_RTM_Cpx .* Win_Doppler.';

            RDM_Complex = fftshift(fft(Current_RTM_Cpx_Win, STFT_FFT_Len, 2), 2);
            RDM_Mag_Stack(:,:,ch) = mat2gray(abs(RDM_Complex));
        end

        % TRGS Bottom-K for RDM
        RDM_Data_Matrix = zeros(Total_Channels, nRange * STFT_FFT_Len);
        for ch = 1:Total_Channels
            img = RDM_Mag_Stack(:,:,ch);
            RDM_Data_Matrix(ch, :) = img(:)';
        end

        [~, rdm_sorted_indices] = TraceRatio_GroupSparse_Selection(...
            RDM_Data_Matrix, TR_Top_K, TR_Lambda, TR_SubspaceDim, TR_Samples);

        selected_indices_RDM = flip(rdm_sorted_indices(end-TR_Top_K+1 : end));

        Reference_RDM = RDM_Mag_Stack(:,:,selected_indices_RDM(1));
        Enhanced_RDM = Reference_RDM;
        for i = 2:length(selected_indices_RDM)
            ch_idx = selected_indices_RDM(i);
            Enhanced_RDM = wfusimg(Enhanced_RDM, RDM_Mag_Stack(:,:,ch_idx), wname, level, wmethod, wmethod);
        end
        
        % -----------------------------------------------------------------
        % 4. Post-processing and saving      
        % -----------------------------------------------------------------
        rtm_filename = fullfile(save_dir_rtm, sprintf('%s_Group_%d.png', name_no_ext, group_count));
        dtm_filename = fullfile(save_dir_dtm, sprintf('%s_Group_%d.png', name_no_ext, group_count));
        rdm_filename = fullfile(save_dir_rdm, sprintf('%s_Group_%d.png', name_no_ext, group_count));
        
        save_image_with_clim(Enhanced_RTM, rtm_filename, Img_Size);
        save_image_with_clim(Enhanced_DTM, dtm_filename, Img_Size);
        save_image_with_clim(Enhanced_RDM, rdm_filename, Img_Size);
        
    end
    
    fclose(fid);
    fprintf('  > Finished %s. Generated %d groups.\n', current_file_name, group_count);
end

fprintf('All files processed successfully with TRGS Bottom-K Logic!\n');

%% Helper Function for Image Saving
function save_image_with_clim(raw_img, filename, target_size)
    % Normalize raw data to [0, 1]
    min_val = min(raw_img(:));
    max_val = max(raw_img(:));
    if max_val == min_val
        norm_img = zeros(size(raw_img));
    else
        norm_img = (raw_img - min_val) / (max_val - min_val);
    end
    
    % Log transform adding epsilon to avoid log(0)
    log_img = log(norm_img + 1e-6);
    
    % Apply cLim [-3, 0]
    clim_min = -3;
    clim_max = 0;
    
    % Clamp data
    log_img(log_img < clim_min) = clim_min;
    log_img(log_img > clim_max) = clim_max;
    
    % Map [-3, 0] to [0, 1] for image representation
    scaled_img = (log_img - clim_min) / (clim_max - clim_min);
    
    % Resize the scalar matrix with interpolation
    resized_scalar = imresize(scaled_img, target_size);
    
    % Apply Jet colormap, convert scalar [0,1] to indices [1, 256]
    idx_img = gray2ind(resized_scalar, 256);
    rgb_img = ind2rgb(idx_img, jet(256));
    
    % Save
    imwrite(rgb_img, filename);
end

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
    max_iter = 10; % Reduced for batch efficiency
    
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