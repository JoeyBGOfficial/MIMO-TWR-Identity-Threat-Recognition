%% Script for Simulated RTM Generation with Compact MIMO UWB Radar
% Original Author: Renming Liu, Yan Tang, Shaoming Zhang, Yusheng Li, and Jianmei Wang.
% Reproduced By: JoeyBG.
% Date: 2025-12-25.
% Affiliation: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description:
%   1 Simulates 2x2 MIMO radar data using original system parameters.
%   2 Supports Walking and GunCarrying activity simulation.
%   3 Generates radar echo signals through-the-wall environment.
%   4 Processes and visualizes Range-Time Maps (RTM) for all 4 channels.

%% Initialization
clear all;
close all;
clc;
disp('---------- © Author: JoeyBG © ----------');

%% Parameter Definitions
% --- Activity Selection ---
activity_type = 'GunCarrying';                                              % Options: Walking or GunCarrying

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
wall_position_xyz = [1.5, 0, 1.2];                                          % Wall Center position
wall_thickness = 0.05;                                                      % Thickness (cm)
wall_epsilon_r = 2.5;                                                       % Dielectric constant
wall_loss_tangent = 0.02;                                                   % Loss tangent

% --- Visualization Parameters ---
Font_Name = 'Palatino Linotype'; 
Font_Size_Basis = 12;
Font_Size_Title = 14;
JoeyBG_Colormap = [0.6196 0.0039 0.2588; 0.8353 0.2431 0.3098; 0.9569 0.4275 0.2627; 0.9922 0.6824 0.3804; 0.9961 0.8784 0.5451; 1.0000 1.0000 0.7490; 0.9020 0.9608 0.5961; 0.6706 0.8667 0.6431; 0.4000 0.7608 0.6471; 0.1961 0.5333 0.7412; 0.3686 0.3098 0.6353];
JoeyBG_Colormap_Flip = flip(JoeyBG_Colormap);

% --- Human Kinematic Model Setup ---
initial_position_xy = [2.5, 0];                                             % Start position (m)
person_height = 1.75;                                                       % Height (m)
v_torso = 0.8;                                                              % Speed (m/s)
walking_angle_deg = 180;                                                    % Walking towards radar
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

% --- MIMO Antenna Configuration ---
d_ant = 7.5e-3; 
radar_center_pos = [0, 0, 1.2]; 

% Define Tx and Rx positions relative to center
% Tx1 Top-Left Tx2 Top-Right Rx1 Bottom-Left Rx2 Bottom-Right
Tx_Pos = [ -d_ant/2, 0, d_ant/2;    d_ant/2, 0, d_ant/2 ];
Rx_Pos = [ -d_ant/2, 0, -d_ant/2;   d_ant/2, 0, -d_ant/2 ];

% Create 4 Channels for 2Tx 2Rx
MIMO_Config = zeros(2, 3, 4); % 2 vectors Tx/Rx x 3 coords x 4 channels

% Populate MIMO Configuration
% Channel 1 Tx1-Rx1
MIMO_Config(1,:,1) = radar_center_pos + Tx_Pos(1,:);
MIMO_Config(2,:,1) = radar_center_pos + Rx_Pos(1,:);
% Channel 2 Tx1-Rx2
MIMO_Config(1,:,2) = radar_center_pos + Tx_Pos(1,:);
MIMO_Config(2,:,2) = radar_center_pos + Rx_Pos(2,:);
% Channel 3 Tx2-Rx1
MIMO_Config(1,:,3) = radar_center_pos + Tx_Pos(2,:);
MIMO_Config(2,:,3) = radar_center_pos + Rx_Pos(1,:);
% Channel 4 Tx2-Rx2
MIMO_Config(1,:,4) = radar_center_pos + Tx_Pos(2,:);
MIMO_Config(2,:,4) = radar_center_pos + Rx_Pos(2,:);

N_channels = 4;

%% Section 1 Human Kinematic and Radar Echo Modeling
% --- 1.1 Scatterer Definition ---
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

if strcmp(activity_type, 'Walking')
    % Arms swinging normally
    scatter_info_arms = {
        'R_Elbow', 'R_Shoulder', [0, 0, -L_arm/2], @(t) A_arm*sin(2*pi*f_gait*t+pi), 0.3;
        'R_Hand', 'R_Elbow', [0, 0, -L_arm/2], @(t) 0, 0.2;
        'L_Elbow', 'L_Shoulder', [0, 0, -L_arm/2], @(t) -A_arm*sin(2*pi*f_gait*t+pi), 0.3;
        'L_Hand', 'L_Elbow', [0, 0, -L_arm/2], @(t) 0, 0.2;
    };
    scatter_info = [scatter_info_base; scatter_info_arms];
elseif strcmp(activity_type, 'GunCarrying')
    % Arms fixed carrying a rifle style object
    rot_upper = @(t) deg2rad(-30); 
    rot_fore  = @(t) deg2rad(-80); 
    scatter_info_arms = {
        'R_Elbow', 'R_Shoulder', [0, 0, -L_arm/2], rot_upper, 0.3;
        'R_Hand', 'R_Elbow', [0, 0, -L_arm/2], rot_fore, 0.2;
        'L_Elbow', 'L_Shoulder', [0, 0, -L_arm/2], rot_upper, 0.3;
        'L_Hand', 'L_Elbow', [0, 0, -L_arm/2], rot_fore, 0.2;
    };
    % Gun specific nodes
    scatter_info_gun = {
        'Gun_Stock', 'R_Hand', [0.1, 0, 0], @(t) 0, 0.8; 
        'Gun_Body', 'Gun_Stock', [0.2, 0, 0], @(t) 0, 1.0; 
        'Gun_Muzzle', 'Gun_Body', [0.3, 0, 0], @(t) 0, 0.5; 
    };
    scatter_info = [scatter_info_base; scatter_info_arms; scatter_info_gun];
end

N_scatter = size(scatter_info, 1);
rcs = cell2mat(scatter_info(:,5)); 

% --- 1.2 Kinematic Trajectory Calculation ---
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

% --- 1.3 Echo Signal Generation ---
fprintf('Generating radar echo signals for %d channels Wall %d...\n', N_channels, enable_wall);

% Wall constants calculation
if enable_wall
    eta0 = 376.73; 
    eta_wall = eta0 / sqrt(wall_epsilon_r);
    Transmission_Factor = (2*eta_wall/(eta_wall+eta0)) * (2*eta0/(eta_wall+eta0));
    alpha = pi * fc / c * sqrt(wall_epsilon_r) * wall_loss_tangent;
end

Raw_Data_MIMO = zeros(numADCSamples, N_pulses, N_channels);

for ch = 1:N_channels
    tx_pos = MIMO_Config(1,:,ch);
    rx_pos = MIMO_Config(2,:,ch);
    Raw_Data_Ch = zeros(numADCSamples, N_pulses);
    
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
                % Simple ray tracing check assuming infinite wall in YZ plane
                if (tx_pos(1) < wall_position_xyz(1) && scatter_pos(1) > wall_position_xyz(1)) 
                     dist_in_wall = wall_thickness;
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
    fprintf('  Channel %d generated\n', ch);
end

% Add AWGN Noise
ref_signal_power = mean(abs(Raw_Data_MIMO(:,:,1)).^2, 'all'); 
noise_power = ref_signal_power / (10^(SNR_dB / 10));
Raw_Data_MIMO_Noisy = Raw_Data_MIMO + sqrt(noise_power/2) * (randn(size(Raw_Data_MIMO)) + 1i * randn(size(Raw_Data_MIMO)));

%% Section 2 RTM Generation and Visualization
fprintf('Processing Range-Time Maps...\n');

N_fft_range = 2^nextpow2(numADCSamples);
Range_Axis = (0:N_fft_range/2-1) * (fs / N_fft_range) * (c / (2 * K));

figure('Name', 'MIMO RTM Visualization', 'Color', 'w', 'Position', [100, 50, 1000, 800]);

% Helper for Log Scale Display
get_log = @(img) log((img - min(img(:))) / (max(img(:)) - min(img(:))) + 1e-6);

for ch = 1:N_channels
    % 1 MTI
    Current_Raw = Raw_Data_MIMO_Noisy(:,:,ch);
    MTI_Data = Current_Raw(:, 2:end) - Current_Raw(:, 1:end-1);
    
    % 2 Range Profile Generation
    Range_Profile_Complex = fft(MTI_Data, N_fft_range, 1);
    RTM_Cpx = Range_Profile_Complex(1:N_fft_range/2, :);
    RTM_Mag = mat2gray(abs(RTM_Cpx)); 
    
    % 3 Visualization
    subplot(2, 2, ch);
    imagesc(t_slow(2:end), Range_Axis, get_log(RTM_Mag)); axis xy;
    set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
    colormap(gca, JoeyBG_Colormap_Flip); 
    colorbar; 
    clim([-4 0]);
    
    % Determine Title based on Tx Rx pair
    if ch == 1, pair_str = 'Tx1-Rx1';
    elseif ch == 2, pair_str = 'Tx1-Rx2';
    elseif ch == 3, pair_str = 'Tx2-Rx1';
    else, pair_str = 'Tx2-Rx2';
    end
    
    title(['RTM Channel ' num2str(ch) ' ' pair_str], 'FontName', Font_Name, 'FontSize', Font_Size_Title);
    xlabel('Time s', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
    ylabel('Range m', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
    ylim([0 6]); % Modified to 0-6m as requested
end

fprintf('Simulation and Visualization Complete.\n');