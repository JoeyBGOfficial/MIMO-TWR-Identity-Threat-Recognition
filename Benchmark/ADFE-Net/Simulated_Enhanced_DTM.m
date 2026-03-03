%% Script for Adaptive Doppler Feature Enhancement Simulation
% Original Author: Longzhen Tang, Shisheng Guo, Jiachen Li, Junda Zhu, Guolong Cui, Lingjiang Kong, and Xiaobo Yang.
% Reproduced By: JoeyBG.
% Date: 2025-12-29.
% Affiliation: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description:
%   1. Simulates radar data for Gun Carrying and Walking activities.
%   2. Implements Two-Pulse Cancellation TPC to preserve limb details.
%   3. Implements Average Cancellation AC to preserve torso details.
%   4. Generates Doppler Time Maps DTM for both clutter suppression methods.
%   5. Performs Adaptive Weighted Fusion to generate an Enhanced Spectrogram.
%   6. Visualizes TPC, AC, and Enhanced Spectrograms for comparison.

%% Initialization
clear all;
close all;
clc;
disp('---------- © Author: JoeyBG © ----------');

%% Parameter Definitions
% --- Activity Selection ---
activity_type = 'Walking';                                                  % Options: Walking or GunCarrying

% --- Radar System Parameters ---
fc = 2.5e9;                                                                 % Radar carrier frequency (Hz)
c = 3e8;                                                                    % Speed of light (m/s)
lambda = c / fc;                                                            % Wavelength (m)
Tp = 40e-6;                                                                 % Pulse width (s)
B = 1e9;                                                                    % Chirp bandwidth (Hz)
K = B / Tp;                                                                 % Chirp rate (Hz/s)
PRF = 200;                                                                  % PRF
fs = 4e6;                                                                   % ADC sampling rate (Hz)

% --- Simulation Time Parameters ---
sim_time = 1.0;                                                             % Total simulation duration (s)
PRT = 1 / PRF;                                                              % Pulse Repetition Time
N_pulses = floor(sim_time * PRF);                                           % Total number of pulses
numADCSamples = floor(fs * Tp);                                             % Samples per chirp

% --- Wall Parameters ---
enable_wall = true;                                                         % Enable wall obstruction
wall_position_xyz = [1, 0, 1.25];                                           % Wall Center position
wall_dimensions_lwh = [0.24, 5, 2.5];                                       % Wall Dimensions
wall_epsilon_r = 6;                                                         % Dielectric constant
wall_loss_tangent = 0.03;                                                   % Loss tangent

% --- Spectrogram Processing Parameters ---
STFT_Win_Size = 20;                                                         % Window size for STFT
STFT_Overlap = 18;                                                          % Overlap size for STFT
Fusion_Bandwidth_Sigma = 10;                                                % Sigma for Gaussian fusion mask

% --- Visualization Parameters ---
Font_Name = 'Palatino Linotype'; 
Font_Size_Basis = 12;
Font_Size_Title = 14;
JoeyBG_Colormap = [0.6196 0.0039 0.2588; 0.8353 0.2431 0.3098; 0.9569 0.4275 0.2627; 0.9922 0.6824 0.3804; 0.9961 0.8784 0.5451; 1.0000 1.0000 0.7490; 0.9020 0.9608 0.5961; 0.6706 0.8667 0.6431; 0.4000 0.7608 0.6471; 0.1961 0.5333 0.7412; 0.3686 0.3098 0.6353];
JoeyBG_Colormap_Flip = flip(JoeyBG_Colormap);

% --- Human Kinematic Model Setup ---
initial_position_xy = [2.5, 0];                                             % Start position (m)
person_height = 1.8;                                                        % Height (m)
v_torso = 0.8;                                                              % Speed (m/s)
walking_angle_deg = 180;                                                    % Direction (deg)
f_gait = 1.2;                                                               % Step frequency (Hz)

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

% --- Antenna Configuration ---
tx_pos = [0, 0, 1.5];
rx_pos = [0, 0.1, 1.5];

%% Section 1. Human Kinematic and Radar Echo Modeling
% 1.1 Scatterer Definition
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

% 1.2 Kinematic Trajectory Calculation
fprintf('Calculating kinematic trajectory for %s...\n', activity_type);
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

% 1.3 Echo Signal Generation
fprintf('Generating radar echo signals (Wall: %d)...\n', enable_wall);

% Wall constants calculation
if enable_wall
    eta0 = 376.73; eta_wall = eta0 / sqrt(wall_epsilon_r);
    Transmission_Factor = (2*eta_wall/(eta_wall+eta0)) * (2*eta0/(eta_wall+eta0));
    alpha = pi * fc / c * sqrt(wall_epsilon_r) * wall_loss_tangent;
    wall_x_front = wall_position_xyz(1) + wall_dimensions_lwh(1)/2;
    wall_x_back = wall_position_xyz(1) - wall_dimensions_lwh(1)/2;
end
Raw_Data = zeros(numADCSamples, N_pulses);

for m = 1:N_pulses
    for i = 1:N_scatter
        scatter_pos = squeeze(pos(i, :, m));
        dist_tx = norm(scatter_pos - tx_pos);
        dist_rx = norm(scatter_pos - rx_pos);
        total_range = dist_tx + dist_rx;
        
        % Radar Equation
        amplitude = sqrt(rcs(i)) / ((total_range/2)^2);
        
        % Apply Wall Attenuation and Delay
        if enable_wall
            if (tx_pos(1) < wall_x_back && scatter_pos(1) > wall_x_front) 
                    dist_in_wall = abs(wall_dimensions_lwh(1)); 
                    amplitude = amplitude * Transmission_Factor * exp(-alpha * dist_in_wall);
                    total_range = total_range + dist_in_wall*(sqrt(wall_epsilon_r)-1);
            end
        end
        
        % Dechirped Signal Formula
        tau = total_range / c;
        beat_signal = amplitude .* exp(-1j * 2 * pi * (fc * tau + K * tau .* t_fast));
        Raw_Data(:, m) = Raw_Data(:, m) + beat_signal';
    end
end

%% Section 2. Range Profile and Clutter Suppression
fprintf('Processing Range Profiles and Clutter Suppression...\n');

N_fft_range = 2^nextpow2(numADCSamples);
Range_Profile_Complex = fft(Raw_Data, N_fft_range, 1);
Range_Profile_Complex = Range_Profile_Complex(1:N_fft_range/2, :);

% 2.1 Two-Pulse Cancellation TPC
% Subtracts consecutive pulses to remove static clutter but may attenuate slow torso motion
RTM_TPC = Range_Profile_Complex(:, 2:end) - Range_Profile_Complex(:, 1:end-1);

% 2.2 Average Cancellation AC
% Subtracts mean over slow time to remove static background while preserving more low-freq body info
Mean_Clutter = mean(Range_Profile_Complex, 2);
RTM_AC = Range_Profile_Complex - Mean_Clutter;
% Align dimensions with TPC for fusion 
RTM_AC = RTM_AC(:, 1:end-1); 

%% Section 3. Spectrogram Generation and Adaptive Fusion
fprintf('Generating Enhanced Doppler Spectrograms...\n');

N_fft_doppler = 512;
STFT_Window = hamming(STFT_Win_Size);

% Function to generate DTM by processing all range bins
generate_DTM = @(RTM_Input) squeeze(sum(abs(stft(RTM_Input.', PRF, ...
    'Window', STFT_Window, 'OverlapLength', STFT_Overlap, 'FFTLength', N_fft_doppler)), 3));

% 3.1 Generate Individual Spectrograms
DTM_TPC = generate_DTM(RTM_TPC);
DTM_AC  = generate_DTM(RTM_AC);

% Get Time and Freq vectors
[~, f_vec, t_vec] = stft(RTM_TPC(1,:), PRF, 'Window', STFT_Window, ...
    'OverlapLength', STFT_Overlap, 'FFTLength', N_fft_doppler);

% Normalize Spectrograms to 0-1 range for fair fusion
DTM_TPC_Norm = DTM_TPC / max(DTM_TPC(:));
DTM_AC_Norm  = DTM_AC / max(DTM_AC(:));

% 3.2 Adaptive Weighted Fusion Simulation
freq_indices = linspace(-PRF/2, PRF/2, N_fft_doppler)';

% Gaussian weight for Low Freq preference AC
Weight_AC = exp(- (freq_indices.^2) / (2 * Fusion_Bandwidth_Sigma^2));
% Inverse weight for High Freq preference TPC
Weight_TPC = 1 - Weight_AC; 

% Expand weights to match time dimension
Weight_AC_Map = repmat(Weight_AC, 1, length(t_vec));
Weight_TPC_Map = repmat(Weight_TPC, 1, length(t_vec));

% Perform Fusion
DTM_Enhanced = (Weight_TPC_Map .* DTM_TPC_Norm) + (Weight_AC_Map .* DTM_AC_Norm);
DTM_Enhanced = DTM_Enhanced / max(DTM_Enhanced(:)); % Re-normalize

%% Section 4. Visualization
fprintf('Visualizing Results...\n');
figure('Name', ['ADFE Spectrogram Analysis - ' activity_type], 'Color', 'w', 'Position', [100, 100, 1400, 500]);

% Plot 1: Two-Pulse Cancellation TPC
subplot(1, 3, 1);
imagesc(t_vec, f_vec, DTM_TPC_Norm);
colormap(JoeyBG_Colormap_Flip);
caxis([0 1]);
title('Two-Pulse Cancellation', 'FontName', Font_Name, 'FontSize', Font_Size_Title);
xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
ylabel('Doppler (Hz)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'YDir', 'normal');
colorbar;

% Plot 2: Average Cancellation AC
subplot(1, 3, 2);
imagesc(t_vec, f_vec, DTM_AC_Norm);
colormap(JoeyBG_Colormap_Flip);
caxis([0 1]);
title('Average Cancellation', 'FontName', Font_Name, 'FontSize', Font_Size_Title);
xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
ylabel('Doppler (Hz)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'YDir', 'normal');
colorbar;

% Plot 3: Doppler Enhanced Spectrogram ADFE
subplot(1, 3, 3);
imagesc(t_vec, f_vec, DTM_Enhanced);
colormap(JoeyBG_Colormap_Flip);
caxis([0 1]);
title('Enhanced Spectrogram', 'FontName', Font_Name, 'FontSize', Font_Size_Title);
xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
ylabel('Doppler (Hz)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'YDir', 'normal');
cb = colorbar;
cb.Label.String = 'Normalized Amplitude';
cb.Label.FontName = Font_Name;
cb.Label.FontSize = Font_Size_Basis;

fprintf('Processing Complete.\n');