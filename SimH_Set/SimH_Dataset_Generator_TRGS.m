%% Script for SimH Dataset Generation: TRGS-Based MIMO Fusion
% Original Author: JoeyBG.
% Improved By: JoeyBG.
% Date: 2025-12-09.
% Affiliation: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description: 
%   Generates synthetic radar datasets (RTM, DTM, RDM) using Trace Ratio Group Sparse (TRGS) fusion.
%   Strictly follows the provided SimH_Datas_Reading_Processing_TRGS.m logic.
%   Outputs 1024x1024 images in 'jet' colormap without axes.

%% Initialization
clear all;
close all;
clc;
rng('shuffle');                                                             % Initialize random generator
disp('---------- © Author: JoeyBG © ----------');

%% Global Parameter Definitions
% --- Radar System Parameters ---
fc = 2.5e9;                                         
c = 3e8;                                            
lambda = c / fc;                                    
Tp = 40e-6;                                         
B = 1e9;                                            
K = B / Tp;                                         
PRF = 200;                                          
fs = 4e6;                                           

% --- Simulation Time Parameters ---
sim_time = 1.0;                                      
PRT = 1 / PRF;                                      
N_pulses = floor(sim_time * PRF);                   
numADCSamples = floor(fs * Tp);                     

% --- Noise Parameter ---
SNR_dB = 10;                                        

% --- Wall Parameters ---
enable_wall = true;                                 
wall_position_xyz_base = [1, 0, 1.25];                   
wall_dimensions_lwh = [0.24, 5, 2.5];               
wall_epsilon_r = 6;                                 
wall_loss_tangent = 0.03;                           

% --- Fusion & TRGS Parameters ---
wname = 'db4';                                      
wmethod = 'mean';                                   
level = 2;                                          
TR_Top_K = 8;                                                               % Select Top 8 Channels
TR_Lambda = 0.1;                                                            % Regularization
TR_SubspaceDim = 5;                                 
TR_Samples = 8192;                                                          % Subsampling for speed

% --- MIMO Antenna Configuration ---
radar_center_pos = [0, 0, 1.5]; 
N_channels = 64;
Antenna_Config = zeros(2, 3, N_channels);
for k = 1:N_channels
    offset_x = (mod(k-1, 8) - 3.5) * 0.1;
    offset_z = (floor((k-1)/8) - 3.5) * 0.1;
    Antenna_Config(1,:,k) = radar_center_pos + [offset_x, 0, offset_z];     
    Antenna_Config(2,:,k) = radar_center_pos + [offset_x, 0.1, offset_z];   
end

%% Dataset Configuration
% Define Output Paths
root_RTM = 'SimH_RTM_Set_TRGS';
root_DTM = 'SimH_DTM_Set_TRGS';
root_RDM = 'SimH_RDM_Set_TRGS';

% Define Classes
classes = struct(...
    'Name', {'P1_Gun', 'P1_Nogun', 'P2_Gun', 'P2_Nogun', ...
             'P3_Gun', 'P3_Nogun', 'P4_Gun', 'P4_Nogun'}, ...
    'Height', {1.8, 1.8, 1.7, 1.7, 1.6, 1.6, 1.5, 1.5}, ...
    'Velocity', {1.2, 1.2, 1.13, 1.13, 1.07, 1.07, 1.0, 1.0}, ...
    'Activity', {'GunCarrying', 'Walking', 'GunCarrying', 'Walking', ...
                 'GunCarrying', 'Walking', 'GunCarrying', 'Walking'}, ...
    'Count', {184, 148, 121, 59, 59, 59, 59, 59} ...
);

% Create Directory Structure
create_dirs(root_RTM, classes);
create_dirs(root_DTM, classes);
create_dirs(root_RDM, classes);

%% Main Generation Loop
total_samples = sum([classes.Count]);
current_sample = 0;

for cls_idx = 1:length(classes)
    cls = classes(cls_idx);
    fprintf('\nProcessing Class: %s (Target: %d samples)\n', cls.Name, cls.Count);
    
    for grp_idx = 1:cls.Count
        current_sample = current_sample + 1;
        fprintf('  [Total Progress: %.1f%%] Generating Group %d/%d for %s...\n', ...
            (current_sample/total_samples)*100, grp_idx, cls.Count, cls.Name);
        
        %% 1. Parameter Randomization
        rand_pos_x = 2 + (rand - 0.5) * 0.5; 
        rand_pos_y = 0 + (rand - 0.5) * 0.5;
        initial_position_xy = [rand_pos_x, rand_pos_y];
        walking_angle_deg = (rand - 0.5) * 30;
        v_torso = cls.Velocity * (1 + (rand - 0.5) * 0.1);
        f_gait = (v_torso / cls.Velocity) * 1.0; 
        
        activity_type = cls.Activity;
        person_height = cls.Height;
        
        %% 2. Kinematic Model
        % Anthropometric Scaling
        h_torso = person_height * (1.2 / 1.8);              
        L_thigh = person_height * (0.45 / 1.8);             
        L_calf = person_height * (0.45 / 1.8);              
        L_arm = person_height * (0.6 / 1.8);                
        head_z_offset = person_height * (0.3 / 1.8);        
        shoulder_z_offset = person_height * (0.2 / 1.8);    
        hip_z_offset = person_height * (-0.3 / 1.8);        
        shoulder_y_offset = 0.2;                            

        % Swing Amplitudes
        A_thigh = deg2rad(30); A_calf = deg2rad(45); A_arm = deg2rad(35);   
        
        % Scatterers
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
        
        % Trajectory
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

        %% 3. Radar Echo Generation
        eta0 = 376.73; eta_wall = eta0 / sqrt(wall_epsilon_r);
        Transmission_Factor = (2*eta_wall/(eta_wall+eta0)) * (2*eta0/(eta_wall+eta0));
        alpha_wall = pi * fc / c * sqrt(wall_epsilon_r) * wall_loss_tangent;
        wall_x_front = wall_position_xyz_base(1) + wall_dimensions_lwh(1)/2;
        wall_x_back = wall_position_xyz_base(1) - wall_dimensions_lwh(1)/2;
        
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
                             amplitude = amplitude * Transmission_Factor * exp(-alpha_wall * dist_in_wall);
                             total_range = total_range + dist_in_wall*(sqrt(wall_epsilon_r)-1);
                        end
                    end
                    
                    tau = total_range / c;
                    beat_signal = amplitude .* exp(-1j * 2 * pi * (fc * tau + K * tau .* t_fast));
                    Raw_Data_Ch(:, m) = Raw_Data_Ch(:, m) + beat_signal';
                end
            end
            Raw_Data_MIMO(:,:,ch) = Raw_Data_Ch;
        end
        
        % Noise
        ref_signal_power = mean(abs(Raw_Data_MIMO(:,:,1)).^2, 'all'); 
        noise_power = ref_signal_power / (10^(SNR_dB / 10));
        Raw_Data_MIMO_Noisy = Raw_Data_MIMO + sqrt(noise_power/2) * (randn(size(Raw_Data_MIMO)) + 1i * randn(size(Raw_Data_MIMO)));

        %% 4. Feature Generation (RTM/DTM/RDM)
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
        RDM_Window = hamming(N_pulses-1)';
        
        RTM_Mag_Stack = zeros(N_fft_range/2, N_pulses-1, N_channels);
        DTM_Mag_Stack = zeros(length(f_stft), length(t_stft), N_channels);
        RDM_Mag_Stack = zeros(N_fft_range/2, N_fft_doppler, N_channels);
        
        for ch = 1:N_channels
            Current_Raw = Raw_Data_MIMO_Noisy(:,:,ch);
            MTI_Data = Current_Raw(:, 2:end) - Current_Raw(:, 1:end-1);
            
            % RTM
            RTM_Cpx = fft(MTI_Data, N_fft_range, 1);
            RTM_Cpx = RTM_Cpx(1:N_fft_range/2, :);
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

        %% 5. TRGS Fusion Logic
        % Create wrapper for the TRGS call
        process_trgs_routine = @(Stack) ...
            run_trgs_and_fuse(Stack, TR_Top_K, TR_Lambda, TR_SubspaceDim, TR_Samples, wname, level, wmethod);
        
        [~, Enh_RTM] = process_trgs_routine(RTM_Mag_Stack);
        [~, Enh_DTM] = process_trgs_routine(DTM_Mag_Stack);
        [~, Enh_RDM] = process_trgs_routine(RDM_Mag_Stack);
        
        %% 6. Image Export
        filename_base = sprintf('%s_Group_%d.png', cls.Name, grp_idx);
        Max_Doppler = PRF / 2;
        Doppler_Axis = linspace(-Max_Doppler, Max_Doppler, N_fft_doppler);
        
        % Save RTM
        save_figure(Enh_RTM, fullfile(root_RTM, cls.Name, filename_base), ...
                    t_slow(2:end), Range_Axis, [0 6], [], 'jet');
        % Save DTM
        save_figure(Enh_DTM, fullfile(root_DTM, cls.Name, filename_base), ...
                    t_stft, f_stft, [-Max_Doppler Max_Doppler], [], 'jet');
        % Save RDM
        save_figure(Enh_RDM, fullfile(root_RDM, cls.Name, filename_base), ...
                    Doppler_Axis, Range_Axis, [0 6], [-Max_Doppler Max_Doppler], 'jet');
        
    end
end
disp('---------- Dataset Generation Complete ----------');

%% Helper Functions
function create_dirs(root, classes)
    if ~exist(root, 'dir'), mkdir(root); end
    for i = 1:length(classes)
        subpath = fullfile(root, classes(i).Name);
        if ~exist(subpath, 'dir'), mkdir(subpath); end
    end
end

function [Ref_Img, Enh_Img] = run_trgs_and_fuse(Stack, TopK, Lambda, SubDim, nSamples, wname, level, wmethod)
    [d1, d2, N_ch] = size(Stack);
    % Flatten: [Features (Channels) x Samples (Pixels)]
    Data_Matrix = zeros(N_ch, d1 * d2);
    for c = 1:N_ch
        img = Stack(:,:,c);
        Data_Matrix(c, :) = img(:)';
    end
    
    % TRGS Selection
    [~, sorted_indices] = TraceRatio_GroupSparse_Selection(Data_Matrix, Lambda, SubDim, nSamples);
    
    % Fusion
    Selected_Indices = sorted_indices(1:min(TopK, N_ch));
    Ref_Img = Stack(:,:,Selected_Indices(1)); 
    Enh_Img = Ref_Img;
    
    if length(Selected_Indices) > 1
        for i = 2:length(Selected_Indices)
            idx = Selected_Indices(i);
            Enh_Img = wfusimg(Enh_Img, Stack(:,:,idx), wname, level, wmethod, wmethod);
        end
    end
end

function [scores, sorted_indices] = TraceRatio_GroupSparse_Selection(X_raw, lambda, m, n_samples)
    [d, N_total] = size(X_raw);
    if N_total > n_samples
        rand_idx = randperm(N_total, n_samples);
        X = X_raw(:, rand_idx);
    else
        X = X_raw;
    end 
    [~, n] = size(X);
    X = X - mean(X, 2);
    
    % Graph Construction
    k = 5; 
    dist_matrix = pdist2(X', X').^2; 
    W_graph = zeros(n, n);
    for i = 1:n
        [~, idx] = sort(dist_matrix(i, :), 'ascend');
        nbs = idx(2:k+1);
        sigma = mean(dist_matrix(i, nbs)); 
        if sigma == 0, sigma = 1e-5; end
        W_graph(i, nbs) = exp(-dist_matrix(i, nbs) / (2*sigma^2));
    end
    W_graph = (W_graph + W_graph') / 2;
    D_graph = diag(sum(W_graph, 2));
    L = D_graph - W_graph;
    
    % Scatter Matrices
    Sw = X * L * X'; 
    Sw = Sw + 1e-4 * eye(d);
    St = X * X';
    Sb = St - Sw;
    
    % Iterative TR Solver
    [U_pca, ~, ~] = svd(X, 'econ');
    if size(U_pca, 2) < m, m = size(U_pca, 2); end
    W = U_pca(:, 1:m);    
    max_iter = 10;
    
    for iter = 1:max_iter
        d_diag = zeros(d, 1);
        for i = 1:d
            wi_norm = norm(W(i, :), 2);
            d_diag(i) = 1 / (2 * wi_norm + 1e-6); 
        end
        D_sparse = diag(d_diag);
        
        num = trace(W' * Sb * W);
        den = trace(W' * (Sw + lambda * D_sparse) * W);
        if den == 0, den = 1e-6; end
        eta = num / den;
        
        P = Sb - eta * (Sw + lambda * D_sparse);
        P = (P + P') / 2;        
        [V, E_val] = eig(P);
        [~, idx] = sort(diag(E_val), 'descend');
        W = V(:, idx(1:m));
        [W, ~] = qr(W, 0);
    end
    
    scores = zeros(d, 1);
    for i = 1:d, scores(i) = norm(W(i, :), 2); end
    [~, sorted_indices] = sort(scores, 'descend');
end

function save_figure(img_data, filepath, x_axis, y_axis, y_lims, x_lims, cmap_name)
    % Log Transform
    log_img = log((img_data - min(img_data(:))) / (max(img_data(:)) - min(img_data(:))) + 1e-6);
    
    % Invisible figure
    f = figure('Visible', 'off', 'Position', [0, 0, 1024, 1024]);
    ax = axes('Position', [0 0 1 1]);
    
    imagesc(x_axis, y_axis, log_img); 
    axis xy;
    colormap(ax, cmap_name);
    clim([-4 0]); 
    
    if ~isempty(y_lims), ylim(y_lims); end
    if ~isempty(x_lims), xlim(x_lims); end
    axis off;
    
    exportgraphics(f, filepath, 'Resolution', 72);
    close(f);
end