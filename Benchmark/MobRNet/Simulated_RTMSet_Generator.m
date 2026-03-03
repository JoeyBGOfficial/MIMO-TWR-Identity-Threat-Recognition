%% Script for Simulated MIMO RTM Dataset Generation
% Original Author: Renming Liu, Yan Tang, Shaoming Zhang, Yusheng Li, and Jianmei Wang.
% Reproduced By: JoeyBG.
% Date: 2025-12-25.
% Affiliation: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description:
%   1. Simulates 2x2 MIMO radar data of 4 Channels based on 'Simulated_RTMs.m'.
%   2. Generates aligned datasets for 8 Classes (P1-P4, Gun/NoGun).
%   3. Saves RTMs as 1024x1024 pure images in separate channel folders.
%   4. Directory Structure:
%      - Simulated_Channel1/ [Class_Name] / [Image].png
%      - ...
%      - Simulated_Channel4/ [Class_Name] / [Image].png

%% Initialization
clear all;
close all;
clc;
disp('---------- © Author: JoeyBG © ----------');

%% Global Parameter Definitions
% --- Radar System Parameters ---
fc = 2.5e9;                                                                 % Radar carrier frequency (Hz)
c = 3e8;                                                                    % Speed of light (m/s)
lambda = c / fc;                                                            % Wavelength (m)
Tp = 40e-6;                                                                 % Pulse width (s)
B = 1e9;                                                                    % Chirp bandwidth (Hz)
K = B / Tp;                                                                 % Chirp rate (Hz/s)
PRF = 200;                                                                  % Pulse Repetition Frequency (Hz)
fs = 4e6;                                                                   % ADC sampling rate (Hz)

% --- Simulation Time Parameters ---
sim_time = 1.0;                                                             % Total simulation duration (s)
PRT = 1 / PRF;                                                              % Pulse Repetition Time (s)
N_pulses = floor(sim_time * PRF);                                           % Total number of pulses
numADCSamples = floor(fs * Tp);                                             % Samples per chirp

% --- Noise Parameter ---
SNR_dB = 15;                                                                % Signal-to-Noise Ratio (dB)

% --- Wall Parameters ---
enable_wall = true;                                                         % Enable wall obstruction
wall_position_xyz_base = [1.5, 0, 1.2];                                     % Wall Center position
wall_thickness = 0.05;                                                      % Thickness (m)
wall_epsilon_r = 2.5;                                                       % Dielectric constant
wall_loss_tangent = 0.02;                                                   % Loss tangent

% --- MIMO Antenna Configuration ---
d_ant = 7.5e-3; 
radar_center_pos = [0, 0, 1.2]; 

% Define Tx and Rx positions relative to center
% Tx1 Top-Left, Tx2 Top-Right, Rx1 Bottom-Left, Rx2 Bottom-Right
Tx_Pos = [ -d_ant/2, 0, d_ant/2;    d_ant/2, 0, d_ant/2 ];
Rx_Pos = [ -d_ant/2, 0, -d_ant/2;   d_ant/2, 0, -d_ant/2 ];

% Create 4 Channels for 2Tx 2Rx
MIMO_Config = zeros(2, 3, 4); 
% Channel 1: Tx1-Rx1
MIMO_Config(1,:,1) = radar_center_pos + Tx_Pos(1,:);
MIMO_Config(2,:,1) = radar_center_pos + Rx_Pos(1,:);
% Channel 2: Tx1-Rx2
MIMO_Config(1,:,2) = radar_center_pos + Tx_Pos(1,:);
MIMO_Config(2,:,2) = radar_center_pos + Rx_Pos(2,:);
% Channel 3: Tx2-Rx1
MIMO_Config(1,:,3) = radar_center_pos + Tx_Pos(2,:);
MIMO_Config(2,:,3) = radar_center_pos + Rx_Pos(1,:);
% Channel 4: Tx2-Rx2
MIMO_Config(1,:,4) = radar_center_pos + Tx_Pos(2,:);
MIMO_Config(2,:,4) = radar_center_pos + Rx_Pos(2,:);

N_channels = 4;

% --- Image Generation Parameters ---
Target_Img_Size = [1024, 1024];                                             % Output Resolution
Max_Range_Display = 6.0;                                                    % Crop RTM to 6 meters
Colormap_Name = 'jet';                                                      % Requested Colormap

%% Dataset Structure Configuration
% Define Output Paths
root_paths = {'Simulated_Channel1', 'Simulated_Channel2', 'Simulated_Channel3', 'Simulated_Channel4'};

% Define Classes
classes = struct(...
    'Name', {'P1_Gun', 'P1_Nogun', 'P2_Gun', 'P2_Nogun', ...
             'P3_Gun', 'P3_Nogun', 'P4_Gun', 'P4_Nogun'}, ...
    'Height', {1.8, 1.8, 1.7, 1.7, 1.6, 1.6, 1.5, 1.5}, ...
    'Velocity', {1.2, 1.2, 1.13, 1.13, 1.07, 1.07, 1.0, 1.0}, ...
    'Activity', {'GunCarrying', 'Walking', 'GunCarrying', 'Walking', ...
                 'GunCarrying', 'Walking', 'GunCarrying', 'Walking'}, ...
    'Count', {368, 296, 242, 118, 118, 118, 118, 118} ...
);

% Create Directory Structure
create_dataset_dirs(root_paths, classes);

%% Main Generation Loop
total_samples = sum([classes.Count]);
global_counter = 0;

for cls_idx = 1:length(classes)
    cls = classes(cls_idx);
    fprintf('\nProcessing Class: %s (Target: %d samples)\n', cls.Name, cls.Count);
    
    for grp_idx = 1:cls.Count
        global_counter = global_counter + 1;
        fprintf('  [Progress: %.2f%%] Class %s | Sample %d/%d\n', ...
            (global_counter/total_samples)*100, cls.Name, grp_idx, cls.Count);
        
        %% 1. Parameter Randomization
        rand_pos_x = 2.5 + (rand - 0.5) * 0.5; 
        rand_pos_y = 0 + (rand - 0.5) * 0.5;
        initial_position_xy = [rand_pos_x, rand_pos_y];
        
        walking_angle_deg = 180 + (rand - 0.5) * 30;
        v_torso = cls.Velocity * (1 + (rand - 0.5) * 0.1);
        f_gait = (v_torso / cls.Velocity) * 1.2; 
        
        current_activity = cls.Activity;
        person_height = cls.Height;
        
        %% 2. Kinematic Trajectory Calculation
        h_torso = person_height * (1.2 / 1.8);              
        L_thigh = person_height * (0.45 / 1.8);             
        L_calf = person_height * (0.45 / 1.8);              
        L_arm = person_height * (0.6 / 1.8);                
        head_z_offset = person_height * (0.3 / 1.8);        
        shoulder_z_offset = person_height * (0.2 / 1.8);    
        hip_z_offset = person_height * (-0.3 / 1.8);        
        shoulder_y_offset = 0.2; 
        
        A_thigh = deg2rad(30); A_calf = deg2rad(45); A_arm = deg2rad(35);
        
        scatter_info_base = {
            'Torso', [], [0, 0, 0], @(t) 0, 1.0;
            'Head', 'Torso', [0, 0, head_z_offset], @(t) 0, 0.5;
            'Hip', 'Torso', [0, 0, hip_z_offset], @(t) 0, 0.3;
            'R_Knee', 'Hip', [0, 0, -L_thigh], @(t) A_thigh*sin(2*pi*f_gait*t), 0.4;
            'R_Ankle', 'R_Knee', [0, 0, -L_calf], @(t) A_calf*sin(2*pi*f_gait*t+pi/4), 0.3;
            'L_Knee', 'Hip', [0, 0, -L_thigh], @(t) -A_thigh*sin(2*pi*f_gait*t), 0.4;
            'L_Ankle', 'L_Knee', [0, 0, -L_calf], @(t) -A_calf*sin(2*pi*f_gait*t+pi/4), 0.3;
            'R_Shoulder', 'Torso', [0, -shoulder_y_offset, shoulder_z_offset], @(t) 0, 0.2;
            'L_Shoulder', 'Torso', [0, shoulder_y_offset, shoulder_z_offset], @(t) 0, 0.2;
        };

        if strcmp(current_activity, 'Walking')
            scatter_info_arms = {
                'R_Elbow', 'R_Shoulder', [0, 0, -L_arm/2], @(t) A_arm*sin(2*pi*f_gait*t+pi), 0.3;
                'R_Hand', 'R_Elbow', [0, 0, -L_arm/2], @(t) 0, 0.2;
                'L_Elbow', 'L_Shoulder', [0, 0, -L_arm/2], @(t) -A_arm*sin(2*pi*f_gait*t+pi), 0.3;
                'L_Hand', 'L_Elbow', [0, 0, -L_arm/2], @(t) 0, 0.2;
            };
            scatter_info = [scatter_info_base; scatter_info_arms];
        elseif strcmp(current_activity, 'GunCarrying')
            rot_upper = @(t) deg2rad(-30); 
            rot_fore  = @(t) deg2rad(-80); 
            scatter_info_arms = {
                'R_Elbow', 'R_Shoulder', [0, 0, -L_arm/2], rot_upper, 0.3;
                'R_Hand', 'R_Elbow', [0, 0, -L_arm/2], rot_fore, 0.2;
                'L_Elbow', 'L_Shoulder', [0, 0, -L_arm/2], rot_upper, 0.3;
                'L_Hand', 'L_Elbow', [0, 0, -L_arm/2], rot_fore, 0.2;
            };
            scatter_info_gun = {
                'Gun_Stock', 'R_Hand', [0.1, 0, 0], @(t) 0, 0.8; 
                'Gun_Body', 'Gun_Stock', [0.2, 0, 0], @(t) 0, 1.0; 
                'Gun_Muzzle', 'Gun_Body', [0.3, 0, 0], @(t) 0, 0.5; 
            };
            scatter_info = [scatter_info_base; scatter_info_arms; scatter_info_gun];
        end

        N_scatter = size(scatter_info, 1);
        rcs = cell2mat(scatter_info(:,5));
        
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
        
        %% 3. Radar Signal Generation & Processing
        eta0 = 376.73; 
        eta_wall = eta0 / sqrt(wall_epsilon_r);
        Transmission_Factor = (2*eta_wall/(eta_wall+eta0)) * (2*eta0/(eta_wall+eta0));
        alpha = pi * fc / c * sqrt(wall_epsilon_r) * wall_loss_tangent;
        
        N_fft_range = 2^nextpow2(numADCSamples);
        Range_Axis = (0:N_fft_range/2-1) * (fs / N_fft_range) * (c / (2 * K));
        
        [~, max_range_idx] = min(abs(Range_Axis - Max_Range_Display));
        
        RTM_Buffer = cell(1, N_channels);
        
        for ch = 1:N_channels
            tx_pos_curr = MIMO_Config(1,:,ch);
            rx_pos_curr = MIMO_Config(2,:,ch);
            Raw_Data_Ch = zeros(numADCSamples, N_pulses);
            
            for m = 1:N_pulses
                for i = 1:N_scatter
                    scatter_pos = squeeze(pos(i, :, m));
                    dist_tx = norm(scatter_pos - tx_pos_curr);
                    dist_rx = norm(scatter_pos - rx_pos_curr);
                    total_range = dist_tx + dist_rx;
                    
                    amplitude = sqrt(rcs(i)) / ((total_range/2)^2);
                    
                    if enable_wall
                        if (tx_pos_curr(1) < wall_position_xyz_base(1) && scatter_pos(1) > wall_position_xyz_base(1)) 
                             dist_in_wall = wall_thickness;
                             amplitude = amplitude * Transmission_Factor * exp(-alpha * dist_in_wall);
                             total_range = total_range + dist_in_wall*(sqrt(wall_epsilon_r)-1);
                        end
                    end
                    
                    tau = total_range / c;
                    beat_signal = amplitude .* exp(-1j * 2 * pi * (fc * tau + K * tau .* t_fast));
                    Raw_Data_Ch(:, m) = Raw_Data_Ch(:, m) + beat_signal';
                end
            end
            
            ref_signal_power = mean(abs(Raw_Data_Ch).^2, 'all');
            noise_power = ref_signal_power / (10^(SNR_dB / 10));
            Raw_Data_Noisy = Raw_Data_Ch + sqrt(noise_power/2) * (randn(size(Raw_Data_Ch)) + 1i * randn(size(Raw_Data_Ch)));
            
            MTI_Data = Raw_Data_Noisy(:, 2:end) - Raw_Data_Noisy(:, 1:end-1);
            
            Range_Profile_Complex = fft(MTI_Data, N_fft_range, 1);
            RTM_Cpx = Range_Profile_Complex(1:N_fft_range/2, :);
            RTM_Mag = abs(RTM_Cpx);
            
            % Crop to 0-6m
            RTM_Cropped = RTM_Mag(1:max_range_idx, :);
            
            % Normalize to 0-1 for image generation
            RTM_Norm = (RTM_Cropped - min(RTM_Cropped(:))) / (max(RTM_Cropped(:)) - min(RTM_Cropped(:)));
            
            RTM_Buffer{ch} = RTM_Norm;
        end
        
        %% 4. Save Images
        for ch = 1:N_channels
            Img_Resized = imresize(RTM_Buffer{ch}, Target_Img_Size);
            RGB_Img = ind2rgb(gray2ind(Img_Resized, 256), jet(256));
            
            filename = sprintf('Sample_%d.png', grp_idx);
            full_path = fullfile(root_paths{ch}, cls.Name, filename);
            imwrite(RGB_Img, full_path);
        end
        
    end
end

disp('Full MIMO RTM Dataset Generation Complete.');

%% Helper Function
function create_dataset_dirs(root_paths, classes)
    for r = 1:length(root_paths)
        root = root_paths{r};
        if ~exist(root, 'dir'), mkdir(root); end
        for c = 1:length(classes)
            subpath = fullfile(root, classes(c).Name);
            if ~exist(subpath, 'dir'), mkdir(subpath); end
        end
    end
end