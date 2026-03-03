%% Script for Radar Human Activity Simulation & MIMO Fusion Based on TRGS Method
% Original Author: JoeyBG.
% Modified By: JoeyBG.
% Date: 2025-12-09.
% Affiliation: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description:
%   1. Simulates MIMO radar data for Gun Carrying/Walking.
%   2. Generates RTM, DTM, and RDM for all channels.
%   3. Applies Trace Ratio - Group Sparse (TRGS) algorithm to rank channel importance.
%   4. Selects the Top-K channels.
%   5. Performs Wavelet-based Image Fusion on selected channels.
%   6. Visualizes Reference vs. Enhanced results.

%% Initialization
clear all;
close all;
clc;
disp('---------- © Author: JoeyBG © ----------');

%% Parameter Definitions
% --- Activity Selection ---
activity_type = 'GunCarrying';                      % Options: 'Walking' or 'GunCarrying'

% --- Radar System Parameters ---
fc = 2.5e9;                                         % Radar carrier frequency (Hz)
c = 3e8;                                            % Speed of light (m/s)
lambda = c / fc;                                    % Wavelength (m)
Tp = 40e-6;                                         % Pulse width (s)
B = 1e9;                                            % Chirp bandwidth (Hz)
K = B / Tp;                                         % Chirp rate (Hz/s)
PRF = 200;                                          % PRF
fs = 4e6;                                           % ADC sampling rate (Hz)

% --- Simulation Time Parameters ---
sim_time = 1.0;                                     % Total simulation duration (s) 
PRT = 1 / PRF;                                      % Pulse Repetition Time
N_pulses = floor(sim_time * PRF);                   % Total number of pulses
numADCSamples = floor(fs * Tp);                     % Samples per chirp

% --- Noise Parameter ---
SNR_dB = 10;                                        % Signal-to-Noise Ratio in dB

% --- Wall Parameters ---
enable_wall = true;                                 % Enable wall obstruction
wall_position_xyz = [1, 0, 1.25];                   % Wall Center position
wall_dimensions_lwh = [0.24, 5, 2.5];               % Wall Dimensions
wall_epsilon_r = 6;                                 % Dielectric constant
wall_loss_tangent = 0.03;                           % Loss tangent

% --- Fusion & TRGS Parameters ---
wname = 'db4';                                      % Wavelet name
wmethod = 'mean';                                   % Fusion method
level = 2;                                          % Decomposition level
TR_Top_K = 8;                                       % Number of TOP channels to select for fusion
TR_Lambda = 0.1;                                    % Regularization parameter for Group Sparsity
TR_SubspaceDim = 5;                                 % Dimension of latent subspace
TR_Samples = 8192;                                  % Subsampling for graph construction

% --- Visualization Parameters ---
Font_Name = 'Palatino Linotype'; 
Font_Size_Basis = 12;
Font_Size_Title = 14;
JoeyBG_Colormap = [0.6196 0.0039 0.2588; 0.8353 0.2431 0.3098; 0.9569 0.4275 0.2627; 0.9922 0.6824 0.3804; 0.9961 0.8784 0.5451; 1.0000 1.0000 0.7490; 0.9020 0.9608 0.5961; 0.6706 0.8667 0.6431; 0.4000 0.7608 0.6471; 0.1961 0.5333 0.7412; 0.3686 0.3098 0.6353];
JoeyBG_Colormap_Flip = flip(JoeyBG_Colormap);

% --- Human Kinematic Model Setup ---
initial_position_xy = [2, 0];                       % Start position (m)
person_height = 1.8;                                % Height (m)
v_torso = 1.0;                                      % Speed (m/s)
walking_angle_deg = 0;                              % Direction
f_gait = 1;                                         % Step frequency (Hz)

% --- Anthropometric Scaling --- 
h_torso = person_height * (1.2 / 1.8);              
L_thigh = person_height * (0.45 / 1.8);             
L_calf = person_height * (0.45 / 1.8);              
L_arm = person_height * (0.6 / 1.8);                
head_z_offset = person_height * (0.3 / 1.8);        
shoulder_z_offset = person_height * (0.2 / 1.8);    
hip_z_offset = person_height * (-0.3 / 1.8);        
shoulder_y_offset = 0.2;                            

% --- Swing Amplitudes --- 
A_thigh = deg2rad(30);                              
A_calf = deg2rad(45);                               
A_arm = deg2rad(35);                                

% --- MIMO Antenna Configuration ---
radar_center_pos = [0, 0, 1.5]; 
N_channels = 64;
Antenna_Config = zeros(2, 3, N_channels);

% Generate a synthetic dense array around center
for k = 1:N_channels
    offset_x = (mod(k-1, 8) - 3.5) * 0.1;
    offset_z = (floor((k-1)/8) - 3.5) * 0.1;
    Antenna_Config(1,:,k) = radar_center_pos + [offset_x, 0, offset_z];     % Tx Position
    Antenna_Config(2,:,k) = radar_center_pos + [offset_x, 0.1, offset_z];   % Rx Position
end

%% Section 1. Human Kinematic and Radar Echo Modeling
% --- 1.1. Scatterer Definition ---
scatter_info_base = {
    'Torso', [], [0, 0, 0], @(t) 0, 1.0;
    'Head', 'Torso', [0, 0, head_z_offset], @(t) 0, 0.5;
    'Hip', 'Torso', [0, 0, hip_z_offset], @(t) 0, 0.1;
    'R_Knee', 'Hip', [0, 0, -L_thigh], @(t) A_thigh*sin(2*pi*f_gait*t), 0.4;
    'R_Ankle', 'R_Knee', [0, 0, -L_calf], @(t) A_calf*sin(2*pi*f_gait*t+pi/4), 0.3;
    'L_Knee', 'Hip', [0, 0, -L_thigh], @(t) -A_thigh*sin(2*pi*f_gait*t), 0.4;
    'L_Ankle', 'L_Knee', [0, 0, -L_calf], @(t) -A_calf*sin(2*pi*f_gait*t+pi/4), 0.3;
    'R_Shoulder', 'Torso', [0, -shoulder_y_offset, shoulder_z_offset], @(t) 0, 0.1;
    'L_Shoulder', 'Torso', [0, shoulder_y_offset, shoulder_z_offset], @(t) 0, 0.1;
};

if strcmp(activity_type, 'Walking')
    scatter_info_arms = {
        'R_Elbow', 'R_Shoulder', [0, 0, -L_arm/2], @(t) A_arm*sin(2*pi*f_gait*t+pi), 0.3;
        'R_Hand', 'R_Elbow', [0, 0, -L_arm/2], @(t) 0, 0.2;
        'L_Elbow', 'L_Shoulder', [0, 0, -L_arm/2], @(t) -A_arm*sin(2*pi*f_gait*t+pi), 0.3;
        'L_Hand', 'L_Elbow', [0, 0, -L_arm/2], @(t) 0, 0.2;
    };
    scatter_info = [scatter_info_base; scatter_info_arms];
elseif strcmp(activity_type, 'GunCarrying')
    rot_upper = @(t) deg2rad(-30); 
    rot_fore  = @(t) deg2rad(-80); 
    scatter_info_arms = {
        'R_Elbow', 'R_Shoulder', [0, 0, -L_arm/2], rot_upper, 0.3;
        'R_Hand', 'R_Elbow', [0, 0, -L_arm/2], rot_fore, 0.2;
        'L_Elbow', 'L_Shoulder', [0, 0, -L_arm/2], rot_upper, 0.3;
        'L_Hand', 'L_Elbow', [0, 0, -L_arm/2], rot_fore, 0.2;
    };
    scatter_info_gun = {
        'Gun_Stock', 'R_Hand', [0.1, 0, 0], @(t) 0, 1.5; 
        'Gun_Body', 'Gun_Stock', [0.2, 0, 0], @(t) 0, 2.0; 
        'Gun_Muzzle', 'Gun_Body', [0.3, 0, 0], @(t) 0, 1.0; 
    };
    scatter_info = [scatter_info_base; scatter_info_arms; scatter_info_gun];
end

N_scatter = size(scatter_info, 1);
rcs = cell2mat(scatter_info(:,5)); 

% --- 1.2. Kinematic Trajectory Calculation ---
fprintf('Calculating kinematic trajectory...\n');
t_slow = (0:N_pulses-1) * PRT;
t_fast = (0:numADCSamples-1) / fs; 
pos = zeros(N_scatter, 3, N_pulses); 

walking_angle_rad = deg2rad(walking_angle_deg);
v_x = v_torso * cos(walking_angle_rad); 
v_y = v_torso * sin(walking_angle_rad); 

for m = 1:N_pulses
    t = t_slow(m);
    pos(1,:,m) = [initial_position_xy(1) + v_x * t, initial_position_xy(2) + v_y * t, h_torso];
    for i = 2:N_scatter
        parent_name = scatter_info{i, 2};
        if ~isempty(parent_name)
            parent_idx = find(strcmp(scatter_info(:,1), parent_name));
            parent_pos = squeeze(pos(parent_idx, :, m));
            link_vec = scatter_info{i, 3};
            theta = scatter_info{i, 4}(t);
            Rot_mat = [cos(theta), 0, sin(theta); 0, 1, 0; -sin(theta), 0, cos(theta)];
            pos(i, :, m) = parent_pos + (Rot_mat * link_vec')';
        end
    end
end

% --- 1.3. Echo Signal Generation ---
fprintf('Generating radar echo signals for %d channels (Wall: %d)...\n', N_channels, enable_wall);

if enable_wall
    eta0 = 376.73; eta_wall = eta0 / sqrt(wall_epsilon_r);
    Transmission_Factor = (2*eta_wall/(eta_wall+eta0)) * (2*eta0/(eta_wall+eta0));
    alpha = pi * fc / c * sqrt(wall_epsilon_r) * wall_loss_tangent;
    wall_x_front = wall_position_xyz(1) + wall_dimensions_lwh(1)/2;
    wall_x_back = wall_position_xyz(1) - wall_dimensions_lwh(1)/2;
end

Raw_Data_MIMO = zeros(numADCSamples, N_pulses, N_channels);

for ch = 1:N_channels
    tx_pos = Antenna_Config(1,:,ch);
    rx_pos = Antenna_Config(2,:,ch);
    Raw_Data_Ch = zeros(numADCSamples, N_pulses);
    
    for m = 1:N_pulses
        for i = 1:N_scatter
            scatter_pos = squeeze(pos(i, :, m));
            dist_tx = norm(scatter_pos - tx_pos);
            dist_rx = norm(scatter_pos - rx_pos);
            total_range = dist_tx + dist_rx;
            
            amplitude = sqrt(rcs(i)) / ((total_range/2)^2);
            
            if enable_wall
                if (tx_pos(1) < wall_x_back && scatter_pos(1) > wall_x_front) 
                     dist_in_wall = abs(wall_dimensions_lwh(1)); 
                     amplitude = amplitude * Transmission_Factor * exp(-alpha * dist_in_wall);
                     total_range = total_range + dist_in_wall*(sqrt(wall_epsilon_r)-1);
                end
            end
            
            tau = total_range / c;
            beat_signal = amplitude .* exp(-1j * 2 * pi * (fc * tau + K * tau .* t_fast));
            Raw_Data_Ch(:, m) = Raw_Data_Ch(:, m) + beat_signal';
        end
    end
    Raw_Data_MIMO(:,:,ch) = Raw_Data_Ch;
    if mod(ch, 10) == 0, fprintf('  Channel %d generated.\n', ch); end
end

% Add AWGN Noise
fprintf('Adding Noise (SNR: %d dB)...\n', SNR_dB);
ref_signal_power = mean(abs(Raw_Data_MIMO(:,:,1)).^2, 'all'); 
noise_power = ref_signal_power / (10^(SNR_dB / 10));
Raw_Data_MIMO_Noisy = Raw_Data_MIMO + sqrt(noise_power/2) * (randn(size(Raw_Data_MIMO)) + 1i * randn(size(Raw_Data_MIMO)));

%% Section 2. Multi-Domain Signal Processing
fprintf('Starting Signal Processing and Feature Generation...\n');

torso_pos_all = squeeze(pos(1, :, :));
if size(torso_pos_all, 1) == 3, torso_pos_all = torso_pos_all'; end
dist_vec = torso_pos_all - radar_center_pos;
R_torso = sqrt(sum(dist_vec.^2, 2)).';
compensation_signal = exp(1j * 4 * pi * R_torso(2:end) / lambda); 

N_fft_range = 2^nextpow2(numADCSamples);
Range_Axis = (0:N_fft_range/2-1) * (fs / N_fft_range) * (c / (2 * K));

DTM_Window_Size = floor(0.1 * PRF);
DTM_Window = hamming(DTM_Window_Size, "periodic");
DTM_Overlap = floor(0.9 * DTM_Window_Size);
N_fft_doppler = 200;

[~, f_stft, t_stft] = stft(zeros(1, N_pulses-1), PRF, 'Window', DTM_Window, 'OverlapLength', DTM_Overlap, 'FFTLength', N_fft_doppler);

RTM_Mag_Stack = zeros(N_fft_range/2, N_pulses-1, N_channels);
DTM_Mag_Stack = zeros(length(f_stft), length(t_stft), N_channels);
RDM_Mag_Stack = zeros(N_fft_range/2, N_fft_doppler, N_channels);
RDM_Window = hamming(N_pulses-1)'; 

for ch = 1:N_channels
    Current_Raw = Raw_Data_MIMO_Noisy(:,:,ch);
    MTI_Data = Current_Raw(:, 2:end) - Current_Raw(:, 1:end-1);
    
    % RTM
    Range_Profile_Complex = fft(MTI_Data, N_fft_range, 1);
    RTM_Cpx = Range_Profile_Complex(1:N_fft_range/2, :);
    RTM_Mag_Stack(:,:,ch) = mat2gray(abs(RTM_Cpx)); 
    
    % DTM
    Range_Sum = sum(RTM_Cpx, 1); 
    Sig_Comp = Range_Sum .* compensation_signal; 
    [S_stft, ~, ~] = stft(Sig_Comp, PRF, 'Window', DTM_Window, 'OverlapLength', DTM_Overlap, 'FFTLength', N_fft_doppler);
    DTM_Mag_Stack(:,:,ch) = mat2gray(abs(S_stft));
    
    % RDM
    RTM_Win = RTM_Cpx .* RDM_Window; 
    RDM_Cpx = fftshift(fft(RTM_Win, N_fft_doppler, 2), 2); 
    RDM_Mag_Stack(:,:,ch) = mat2gray(abs(RDM_Cpx));
end

%% Section 3. TRGS Feature Selection and Fusion
fprintf('Starting TRGS Feature Selection and Fusion...\n');

% Create function handle for reusability
process_trgs_routine = @(Stack, K, L, SubDim, Samp, Name) ...
    run_trgs_and_fuse(Stack, K, L, SubDim, Samp, Name, wname, level, wmethod);

% Execute TRGS for all 3 domains
[Ref_RTM, Enh_RTM, Indices_RTM] = process_trgs_routine(RTM_Mag_Stack, TR_Top_K, TR_Lambda, TR_SubspaceDim, TR_Samples, 'RTM');
[Ref_DTM, Enh_DTM, Indices_DTM] = process_trgs_routine(DTM_Mag_Stack, TR_Top_K, TR_Lambda, TR_SubspaceDim, TR_Samples, 'DTM');
[Ref_RDM, Enh_RDM, Indices_RDM] = process_trgs_routine(RDM_Mag_Stack, TR_Top_K, TR_Lambda, TR_SubspaceDim, TR_Samples, 'RDM');

%% Section 4. Visualization
fprintf('Visualizing Results in 3x2 Grid...\n');
figure('Name', 'Simulated TRGS-Based MIMO Fusion', 'Color', 'w', 'Position', [100, 50, 1200, 900]);

get_log = @(img) log((img - min(img(:))) / (max(img(:)) - min(img(:))) + 1e-6);
Max_Doppler = PRF / 2; 

% --- ROW 1: RTM ---
subplot(3, 2, 1);
imagesc(t_slow(2:end), Range_Axis, get_log(Ref_RTM)); axis xy;
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
colormap(gca, JoeyBG_Colormap_Flip); colorbar; clim([-4 0]);
title(['Ref RTM (Top-1: Ch ', num2str(Indices_RTM(1)), ')'], 'FontName', Font_Name, 'FontSize', Font_Size_Title);
xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
ylabel('Range (m)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
ylim([0 6]);

subplot(3, 2, 2);
imagesc(t_slow(2:end), Range_Axis, get_log(Enh_RTM)); axis xy;
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
colormap(gca, JoeyBG_Colormap_Flip); colorbar; clim([-4 0]);
title(['Enhanced RTM (Top-', num2str(TR_Top_K), ' TRGS)'], 'FontName', Font_Name, 'FontSize', Font_Size_Title);
xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
ylabel('Range (m)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
ylim([0 6]);

% --- ROW 2: DTM ---
subplot(3, 2, 3);
imagesc(t_stft, f_stft, get_log(Ref_DTM)); axis xy;
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
colormap(gca, JoeyBG_Colormap_Flip); colorbar; clim([-4 0]);
title(['Ref DTM (Top-1: Ch ', num2str(Indices_DTM(1)), ')'], 'FontName', Font_Name, 'FontSize', Font_Size_Title);
xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
ylabel('Doppler (Hz)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
ylim([-Max_Doppler Max_Doppler]);

subplot(3, 2, 4);
imagesc(t_stft, f_stft, get_log(Enh_DTM)); axis xy;
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
colormap(gca, JoeyBG_Colormap_Flip); colorbar; clim([-4 0]);
title(['Enhanced DTM (Top-', num2str(TR_Top_K), ' TRGS)'], 'FontName', Font_Name, 'FontSize', Font_Size_Title);
xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
ylabel('Doppler (Hz)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
ylim([-Max_Doppler Max_Doppler]);

% --- ROW 3: RDM ---
Doppler_Axis = linspace(-Max_Doppler, Max_Doppler, N_fft_doppler);
subplot(3, 2, 5);
imagesc(Doppler_Axis, Range_Axis, get_log(Ref_RDM)); axis xy;
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
colormap(gca, JoeyBG_Colormap_Flip); colorbar; clim([-4 0]);
title(['Ref RDM (Top-1: Ch ', num2str(Indices_RDM(1)), ')'], 'FontName', Font_Name, 'FontSize', Font_Size_Title);
xlabel('Doppler (Hz)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
ylabel('Range (m)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
ylim([0 6]); xlim([-Max_Doppler Max_Doppler]);

subplot(3, 2, 6);
imagesc(Doppler_Axis, Range_Axis, get_log(Enh_RDM)); axis xy;
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
colormap(gca, JoeyBG_Colormap_Flip); colorbar; clim([-4 0]);
title(['Enhanced RDM (Top-', num2str(TR_Top_K), ' TRGS)'], 'FontName', Font_Name, 'FontSize', Font_Size_Title);
xlabel('Doppler (Hz)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
ylabel('Range (m)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
ylim([0 6]); xlim([-Max_Doppler Max_Doppler]);

fprintf('Processing Complete.\n');

%% Helper Functions
function [Ref_Img, Enh_Img, Selected_Indices] = run_trgs_and_fuse(Stack, TopK, Lambda, SubDim, nSamples, Name, wname, level, wmethod)
    [d1, d2, N_ch] = size(Stack);
    % 1. Flatten Data for TRGS: [Features (Channels) x Samples (Pixels)]
    Data_Matrix = zeros(N_ch, d1 * d2);
    for c = 1:N_ch
        img = Stack(:,:,c);
        Data_Matrix(c, :) = img(:)';
    end
    
    % 2. Run TRGS Selection
    fprintf('  > Calculating TRGS Scores for %s...\n', Name);
    [scores, sorted_indices] = TraceRatio_GroupSparse_Selection(Data_Matrix, Lambda, SubDim, nSamples);
    
    % 3. Select TOP-K
    Selected_Indices = sorted_indices(1:min(TopK, N_ch));
    fprintf('  > %s Top %d Channels Selected: %s\n', Name, length(Selected_Indices), num2str(Selected_Indices'));
    
    % 4. Fusion
    Ref_Img = Stack(:,:,Selected_Indices(1)); % Best channel
    Enh_Img = Ref_Img;
    
    if length(Selected_Indices) > 1
        for i = 2:length(Selected_Indices)
            idx = Selected_Indices(i);
            Enh_Img = wfusimg(Enh_Img, Stack(:,:,idx), wname, level, wmethod, wmethod);
        end
    end
end

function [scores, sorted_indices] = TraceRatio_GroupSparse_Selection(X_raw, lambda, m, n_samples)
    % Inputs:
    %   X_raw: Data matrix [d_features x N_samples] 
    %   lambda: Regularization parameter for Group Sparsity
    %   m: Dimension of the subspace
    %   n_samples: Number of samples to use for graph construction
    % Outputs:
    %   scores: Weight for each feature
    %   sorted_indices: Indices of features sorted by importance
    [d, N_total] = size(X_raw);
    
    % Data subsampling if needed
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
    
    % Calculate pairwise Euclidean distances between samples
    dist_matrix = pdist2(X', X').^2; 
    
    % Build adjacency matrix W_graph
    W_graph = zeros(n, n);
    for i = 1:n
        [~, idx] = sort(dist_matrix(i, :), 'ascend');
        % Connect k neighbors
        nbs = idx(2:k+1);
        % Heat kernel weighting
        sigma = mean(dist_matrix(i, nbs)); 
        if sigma == 0, sigma = 1e-5; end
        W_graph(i, nbs) = exp(-dist_matrix(i, nbs) / (2*sigma^2));
    end
    W_graph = (W_graph + W_graph') / 2; % Symmetrize
    D_graph = diag(sum(W_graph, 2)); % Degree matrix
    L = D_graph - W_graph; % Laplacian matrix
    
    % Calculate scatter matrices
    Sw = X * L * X'; % Within-class scatter
    Sw = Sw + 1e-4 * eye(d); % Regularize Sw
    
    % Total scatter
    St = X * X';
    
    % Between-class scatter
    Sb = St - Sw;
    
    % Iterative TR algorithm with L2,1 Norm
    [U_pca, ~, ~] = svd(X, 'econ');
    if size(U_pca, 2) < m
        m = size(U_pca, 2); 
    end
    W = U_pca(:, 1:m);    
    max_iter = 10; % Reduced slightly for speed
    
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
        
        % Solve Generalized Eigenvalue Problem
        P = Sb - eta * (Sw + lambda * D_sparse);
        P = (P + P') / 2;        
        [V, E_val] = eig(P);
        [~, idx] = sort(diag(E_val), 'descend');
        W = V(:, idx(1:m));
        
        % Orthogonalize W
        [W, ~] = qr(W, 0);
    end
    
    % Feature scoring based on L2 norm of projection rows
    scores = zeros(d, 1);
    for i = 1:d
        scores(i) = norm(W(i, :), 2);
    end
    
    % Sort Descending
    [~, sorted_indices] = sort(scores, 'descend');
end