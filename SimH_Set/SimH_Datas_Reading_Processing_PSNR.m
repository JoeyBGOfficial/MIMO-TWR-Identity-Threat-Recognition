%% Script for Radar Human Activity Simulation & MIMO Fusion Based on PSNR Method
% Original Author: JoeyBG.
% Improved By: JoeyBG.
% Date: 2025-12-09.
% Affiliation: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description:
%   1. Simulates MIMO radar data for Gun Carrying/Walking.
%   2. Generates RTM, Perfectly Compensated DTM, and RDM for all channels.
%   3. Applies Entropy-Minimization and PSNR-Screening to select optimal channels.
%   4. Performs Wavelet-based Image Fusion to enhance features.
%   5. Visualizes Reference vs. Enhanced results in a 3x2 grid.

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

% --- Fusion Parameters ---
wname = 'db4';                                      % Wavelet name
wmethod = 'mean';                                   % Fusion method
level = 2;                                          % Decomposition level
PSNR_Threshold_RTM = 24;                            % PSNR Threshold for RTM
PSNR_Threshold_DTM = 20;                            % PSNR Threshold for DTM
PSNR_Threshold_RDM = 32;                            % PSNR Threshold for RDM

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
% Define body parts, parent nodes, offsets, rotation functions, and RCS
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
    % Arms swinging
    scatter_info_arms = {
        'R_Elbow', 'R_Shoulder', [0, 0, -L_arm/2], @(t) A_arm*sin(2*pi*f_gait*t+pi), 0.3;
        'R_Hand', 'R_Elbow', [0, 0, -L_arm/2], @(t) 0, 0.2;
        'L_Elbow', 'L_Shoulder', [0, 0, -L_arm/2], @(t) -A_arm*sin(2*pi*f_gait*t+pi), 0.3;
        'L_Hand', 'L_Elbow', [0, 0, -L_arm/2], @(t) 0, 0.2;
    };
    scatter_info = [scatter_info_base; scatter_info_arms];
elseif strcmp(activity_type, 'GunCarrying')
    % Arms fixed carrying a gun
    rot_upper = @(t) deg2rad(-30); 
    rot_fore  = @(t) deg2rad(-80); 
    scatter_info_arms = {
        'R_Elbow', 'R_Shoulder', [0, 0, -L_arm/2], rot_upper, 0.3;
        'R_Hand', 'R_Elbow', [0, 0, -L_arm/2], rot_fore, 0.2;
        'L_Elbow', 'L_Shoulder', [0, 0, -L_arm/2], rot_upper, 0.3;
        'L_Hand', 'L_Elbow', [0, 0, -L_arm/2], rot_fore, 0.2;
    };
    % Gun nodes
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
    % Update global torso position
    pos(1,:,m) = [initial_position_xy(1) + v_x * t, initial_position_xy(2) + v_y * t, h_torso];
    
    % Update limbs recursively
    for i = 2:N_scatter
        parent_name = scatter_info{i, 2};
        if ~isempty(parent_name)
            parent_idx = find(strcmp(scatter_info(:,1), parent_name));
            parent_pos = squeeze(pos(parent_idx, :, m));
            link_vec = scatter_info{i, 3};
            theta = scatter_info{i, 4}(t);
            % Rotation Matrix
            Rot_mat = [cos(theta), 0, sin(theta); 0, 1, 0; -sin(theta), 0, cos(theta)];
            pos(i, :, m) = parent_pos + (Rot_mat * link_vec')';
        end
    end
end

% --- 1.3. Echo Signal Generation ---
fprintf('Generating radar echo signals for %d channels (Wall: %d)...\n', N_channels, enable_wall);

% Wall constants calculation
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
            
            % Radar Equation
            amplitude = sqrt(rcs(i)) / ((total_range/2)^2);
            
            % Apply Wall Attenuation & Delay
            if enable_wall
                if (tx_pos(1) < wall_x_back && scatter_pos(1) > wall_x_front) 
                     dist_in_wall = abs(wall_dimensions_lwh(1)); % Approximation
                     amplitude = amplitude * Transmission_Factor * exp(-alpha * dist_in_wall);
                     total_range = total_range + dist_in_wall*(sqrt(wall_epsilon_r)-1);
                end
            end
            
            % Dechirped Signal Formula
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

%% Section 2. Multi-Domain Signal Processing (RTM / DTM / RDM)
fprintf('Starting Signal Processing and Feature Generation...\n');

% --- 2.1 Calculate Compensation Phase ---
torso_pos_all = squeeze(pos(1, :, :));
if size(torso_pos_all, 1) == 3, torso_pos_all = torso_pos_all'; end
dist_vec = torso_pos_all - radar_center_pos;
R_torso = sqrt(sum(dist_vec.^2, 2)).';
compensation_signal = exp(1j * 4 * pi * R_torso(2:end) / lambda); % Used to center the Doppler spectrum

% Processing Parameters
N_fft_range = 2^nextpow2(numADCSamples);
Range_Axis = (0:N_fft_range/2-1) * (fs / N_fft_range) * (c / (2 * K));

% DTM Parameters
DTM_Window_Size = floor(0.1 * PRF);
DTM_Window = hamming(DTM_Window_Size, "periodic");
DTM_Overlap = floor(0.9 * DTM_Window_Size);
N_fft_doppler = 200;

% Pre-run STFT to get axes
[~, f_stft, t_stft] = stft(zeros(1, N_pulses-1), PRF, 'Window', DTM_Window, 'OverlapLength', DTM_Overlap, 'FFTLength', N_fft_doppler);

% Pre-allocate memory
RTM_Mag_Stack = zeros(N_fft_range/2, N_pulses-1, N_channels);
DTM_Mag_Stack = zeros(length(f_stft), length(t_stft), N_channels);
RDM_Mag_Stack = zeros(N_fft_range/2, N_fft_doppler, N_channels);
RDM_Window = hamming(N_pulses-1)'; % Window for RDM slow time dimension

for ch = 1:N_channels
    % 1. MTI
    Current_Raw = Raw_Data_MIMO_Noisy(:,:,ch);
    MTI_Data = Current_Raw(:, 2:end) - Current_Raw(:, 1:end-1);
    
    % 2. RTM Generation
    Range_Profile_Complex = fft(MTI_Data, N_fft_range, 1);
    RTM_Cpx = Range_Profile_Complex(1:N_fft_range/2, :);
    RTM_Mag_Stack(:,:,ch) = mat2gray(abs(RTM_Cpx)); % Store Normalized Magnitude
    
    % 3. DTM Generation
    Range_Sum = sum(RTM_Cpx, 1); % Aggregate energy along range
    Sig_Comp = Range_Sum .* compensation_signal; % Apply Phase Compensation
    [S_stft, ~, ~] = stft(Sig_Comp, PRF, 'Window', DTM_Window, 'OverlapLength', DTM_Overlap, 'FFTLength', N_fft_doppler);
    DTM_Mag_Stack(:,:,ch) = mat2gray(abs(S_stft));
    
    % 4. RDM Generation
    RTM_Win = RTM_Cpx .* RDM_Window; % Apply window along slow time
    RDM_Cpx = fftshift(fft(RTM_Win, N_fft_doppler, 2), 2); % FFT along slow time, shift to center 0 Hz
    RDM_Mag_Stack(:,:,ch) = mat2gray(abs(RDM_Cpx));
end

%% Section 3. Fusion Logic
fprintf('Starting Fusion Process...\n');

% Create an anonymous function to pass the fixed parameters
fusion_routine = @(Stack, Thresh, Name) process_fusion(Stack, Thresh, Name, wname, level, wmethod);

% Execute Fusion for all 3 domains
[Ref_RTM, Enh_RTM, Ref_Idx_RTM] = fusion_routine(RTM_Mag_Stack, PSNR_Threshold_RTM, 'RTM');
[Ref_DTM, Enh_DTM, Ref_Idx_DTM] = fusion_routine(DTM_Mag_Stack, PSNR_Threshold_DTM, 'DTM');
[Ref_RDM, Enh_RDM, Ref_Idx_RDM] = fusion_routine(RDM_Mag_Stack, PSNR_Threshold_RDM, 'RDM');

%% Section 4. Visualization
fprintf('Visualizing Results in 3x2 Grid...\n');
figure('Name', 'Simulated PSNR-Based MIMO Fusion', 'Color', 'w', 'Position', [100, 50, 1200, 900]);

% Helper for Log-Scale Display
get_log = @(img) log((img - min(img(:))) / (max(img(:)) - min(img(:))) + 1e-6);

% Max Doppler Limit based on PRF
Max_Doppler = PRF / 2; 

% --- ROW 1: RTM ---
subplot(3, 2, 1);
imagesc(t_slow(2:end), Range_Axis, get_log(Ref_RTM)); axis xy;
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis); % Apply Font Settings to Axes
colormap(gca, JoeyBG_Colormap_Flip); colorbar; clim([-4 0]);
title(['Ref RTM (Ch ', num2str(Ref_Idx_RTM), ')'], 'FontName', Font_Name, 'FontSize', Font_Size_Title);
xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
ylabel('Range (m)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
ylim([0 6]);

subplot(3, 2, 2);
imagesc(t_slow(2:end), Range_Axis, get_log(Enh_RTM)); axis xy;
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
colormap(gca, JoeyBG_Colormap_Flip); colorbar; clim([-4 0]);
title('Enhanced RTM', 'FontName', Font_Name, 'FontSize', Font_Size_Title);
xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
ylabel('Range (m)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
ylim([0 6]);

% --- ROW 2: DTM ---
subplot(3, 2, 3);
imagesc(t_stft, f_stft, get_log(Ref_DTM)); axis xy;
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
colormap(gca, JoeyBG_Colormap_Flip); colorbar; clim([-4 0]);
title(['Ref DTM (Ch ', num2str(Ref_Idx_DTM), ')'], 'FontName', Font_Name, 'FontSize', Font_Size_Title);
xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
ylabel('Doppler (Hz)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
ylim([-Max_Doppler Max_Doppler]);

subplot(3, 2, 4);
imagesc(t_stft, f_stft, get_log(Enh_DTM)); axis xy;
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
colormap(gca, JoeyBG_Colormap_Flip); colorbar; clim([-4 0]);
title('Enhanced DTM', 'FontName', Font_Name, 'FontSize', Font_Size_Title);
xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
ylabel('Doppler (Hz)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
ylim([-Max_Doppler Max_Doppler]);

% --- ROW 3: RDM ---
Doppler_Axis = linspace(-Max_Doppler, Max_Doppler, N_fft_doppler);
subplot(3, 2, 5);
imagesc(Doppler_Axis, Range_Axis, get_log(Ref_RDM)); axis xy;
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
colormap(gca, JoeyBG_Colormap_Flip); colorbar; clim([-4 0]);
title(['Ref RDM (Ch ', num2str(Ref_Idx_RDM), ')'], 'FontName', Font_Name, 'FontSize', Font_Size_Title);
xlabel('Doppler (Hz)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
ylabel('Range (m)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
ylim([0 6]); xlim([-Max_Doppler Max_Doppler]);

subplot(3, 2, 6);
imagesc(Doppler_Axis, Range_Axis, get_log(Enh_RDM)); axis xy;
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
colormap(gca, JoeyBG_Colormap_Flip); colorbar; clim([-4 0]);
title('Enhanced RDM', 'FontName', Font_Name, 'FontSize', Font_Size_Title);
xlabel('Doppler (Hz)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
ylabel('Range (m)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
ylim([0 6]); xlim([-Max_Doppler Max_Doppler]);

fprintf('Processing Complete.\n');

%% Helper Functions
function [Ref_Img, Enh_Img, Ref_Idx] = process_fusion(Stack, PSNR_Thresh, DomainName, wname, level, wmethod)
    % Function to process Reference Selection, PSNR Screening, and Wavelet Fusion
    % Inputs:
    %   Stack: 3D Matrix of images [Dim1, Dim2, Channels]
    %   PSNR_Thresh: Threshold for selecting channels
    %   DomainName: String for logging
    %   wname: Wavelet name
    %   level: Decomposition level
    %   wmethod: Fusion rule
    [~, ~, N_ch] = size(Stack);
    
    % 1. Entropy Calculation
    entropies = zeros(1, N_ch);
    for c = 1:N_ch
        % Convert to uint8 to calculate histogram entropy properly
        entropies(c) = entropy(im2uint8(Stack(:,:,c)));
    end
    
    % 2. Reference Selection
    [min_ent, Ref_Idx] = min(entropies);
    Ref_Img = Stack(:,:,Ref_Idx);
    fprintf('  > %s Ref Channel: %d (Entropy: %.4f)\n', DomainName, Ref_Idx, min_ent);
    
    % 3. PSNR Screening
    selected_indices = [];
    for c = 1:N_ch
        if c == Ref_Idx, continue; end
        p_val = psnr(Stack(:,:,c), Ref_Img);
        if p_val > PSNR_Thresh
            selected_indices = [selected_indices, c];
        end
    end
    fprintf('  > %s Fusion Candidates: %d / %d\n', DomainName, length(selected_indices), N_ch-1);
    
    % 4. Wavelet Fusion
    Enh_Img = Ref_Img;
    for idx = selected_indices
        % Perform image fusion using Wavelet Toolbox
        Enh_Img = wfusimg(Enh_Img, Stack(:,:,idx), wname, level, wmethod, wmethod);
    end
end