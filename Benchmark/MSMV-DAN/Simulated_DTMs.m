%% Script for DTM Generation with Multistatic MIMO Bio Radar
% Original Author: Yimeng Zhao, Yong Jia, Dong Huang, Li Zhang, Yao Zheng, Jianqi Wang, and Fugui Qi.
% Reproduced By: JoeyBG.
% Date: 2025-12-28.
% Affiliation: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description:
%   1 Simulates radar data for Walking activity using a MIMO array.
%   2 Array configuration includes 2 Tx and 4 Rx antennas.
%   3 Generates 8 channels of raw radar echo data.
%   4 Performs Range Doppler processing to extract DTMs.
%   5 Visualizes the 8 channel DTMs using normalized amplitude.

%% Initialization
clear all;
close all;
clc;
disp('---------- © Author: JoeyBG © ----------');

%% Parameter Definitions
% --- Activity Selection ---
activity_type = 'Walking';

% --- Radar System Parameters ---
fc = 2.5e9;                                                                 % Radar carrier frequency (Hz)
c = 3e8;                                                                    % Speed of light (m/s)
lambda = c / fc;                                                            % Wavelength (m)
Tp = 40e-6;                                                                 % Pulse width (s)
B = 1e9;                                                                    % Chirp bandwidth (Hz)
K = B / Tp;                                                                 % Chirp rate (Hz/s)
PRF = 200;                                                                  % Pulse Repetition Frequency
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

% --- Spectrogram Parameters ---
STFT_Win_Size = 20;                                                         % Window size for STFT
STFT_Overlap = 18;                                                          % Overlap size for STFT

% --- Visualization Parameters ---
Font_Name = 'Palatino Linotype'; 
Font_Size_Basis = 10;
Font_Size_Title = 12;
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

% --- MIMO Array Configuration ---
ant_spacing = 0.6;
% Positions relative to center: T1 R1 R2 R3 R4 T2
y_pos_list = [-1.5, -0.9, -0.3, 0.3, 0.9, 1.5] * ant_spacing;
array_z = 1.5;

tx_locs = [0, y_pos_list(1), array_z;   % T1
           0, y_pos_list(6), array_z];  % T2
           
rx_locs = [0, y_pos_list(2), array_z;   % R1
           0, y_pos_list(3), array_z;   % R2
           0, y_pos_list(4), array_z;   % R3
           0, y_pos_list(5), array_z];  % R4

% Define channel pairs Tx Rx
channels = [1 1; 1 2; 1 3; 1 4; 2 1; 2 2; 2 3; 2 4]; 
num_channels = size(channels, 1);

%% Section 1 Human Kinematic and Radar Echo Modeling
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

% Arms swinging
scatter_info_arms = {
    'R_Elbow', 'R_Shoulder', [0, 0, -L_arm/2], @(t) A_arm*sin(2*pi*f_gait*t+pi), 0.3;
    'R_Hand', 'R_Elbow', [0, 0, -L_arm/2], @(t) 0, 0.2;
    'L_Elbow', 'L_Shoulder', [0, 0, -L_arm/2], @(t) -A_arm*sin(2*pi*f_gait*t+pi), 0.3;
    'L_Hand', 'L_Elbow', [0, 0, -L_arm/2], @(t) 0, 0.2;
};
scatter_info = [scatter_info_base; scatter_info_arms];

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

% 1.3 MIMO Echo Signal Generation
fprintf('Generating 8-channel MIMO radar echo signals...\n');

% Wall constants calculation
if enable_wall
    eta0 = 376.73; eta_wall = eta0 / sqrt(wall_epsilon_r);
    Transmission_Factor = (2*eta_wall/(eta_wall+eta0)) * (2*eta0/(eta_wall+eta0));
    alpha = pi * fc / c * sqrt(wall_epsilon_r) * wall_loss_tangent;
    wall_x_front = wall_position_xyz(1) + wall_dimensions_lwh(1)/2;
    wall_x_back = wall_position_xyz(1) - wall_dimensions_lwh(1)/2;
end

% Data container FastTime x SlowTime x Channel
Raw_Data_MIMO = zeros(numADCSamples, N_pulses, num_channels);

for k = 1:num_channels
    tx_idx = channels(k, 1);
    rx_idx = channels(k, 2);
    
    curr_tx_pos = tx_locs(tx_idx, :);
    curr_rx_pos = rx_locs(rx_idx, :);
    
    fprintf('  Processing Channel %d Tx%d-Rx%d ...\n', k, tx_idx, rx_idx);
    
    for m = 1:N_pulses
        for i = 1:N_scatter
            scatter_pos = squeeze(pos(i, :, m));
            dist_tx = norm(scatter_pos - curr_tx_pos);
            dist_rx = norm(scatter_pos - curr_rx_pos);
            total_range = dist_tx + dist_rx;
            
            % Radar Equation
            amplitude = sqrt(rcs(i)) / ((total_range/2)^2);
            
            % Apply Wall Attenuation and Delay
            if enable_wall
                % Check if path crosses wall range
                if (curr_tx_pos(1) < wall_x_back && scatter_pos(1) > wall_x_front) 
                        dist_in_wall = abs(wall_dimensions_lwh(1)); 
                        amplitude = amplitude * Transmission_Factor * exp(-alpha * dist_in_wall);
                        total_range = total_range + dist_in_wall*(sqrt(wall_epsilon_r)-1);
                end
            end
            
            % Dechirped Signal Formula
            tau = total_range / c;
            beat_signal = amplitude .* exp(-1j * 2 * pi * (fc * tau + K * tau .* t_fast));
            Raw_Data_MIMO(:, m, k) = Raw_Data_MIMO(:, m, k) + beat_signal';
        end
    end
end

%% Section 2 DTM Generation and Visualization
fprintf('Processing DTMs and Visualizing...\n');

N_fft_doppler = 256;
STFT_Window = hamming(STFT_Win_Size);
figure('Name', '8-Channel Multistatic DTMs', 'Color', 'w', 'Position', [50, 50, 1400, 700]);
tiledlayout(2, 4, 'Padding', 'compact', 'TileSpacing', 'compact');

for k = 1:num_channels
    % 2.1 Range Profile and MTI per channel
    raw_k = Raw_Data_MIMO(:, :, k);
    N_fft_range = 2^nextpow2(numADCSamples);
    
    % FFT along fast time
    Range_Profile_Complex = fft(raw_k, N_fft_range, 1);
    Range_Profile_Complex = Range_Profile_Complex(1:N_fft_range/2, :);
    
    % Wall Clutter Removal via MTI
    RTM_Clutter_Removed = Range_Profile_Complex(:, 2:end) - Range_Profile_Complex(:, 1:end-1);
    
    % 2.2 Sum along range bins to get time series for DTM
    dtm_time_series = sum(RTM_Clutter_Removed, 1);
    
    % 2.3 STFT
    [S_stft, f_vec, t_vec] = stft(dtm_time_series, PRF, 'Window', STFT_Window, 'OverlapLength', STFT_Overlap, 'FFTLength', N_fft_doppler);
    
    % Normalize Amplitude
    Mag_DTM = abs(S_stft);
    Norm_DTM = Mag_DTM / max(Mag_DTM(:));
    
    % 2.4 Visualization
    nexttile;
    imagesc(t_vec, f_vec, Norm_DTM);
    colormap(JoeyBG_Colormap_Flip);
    axis xy;
    
    % Styling
    title(['View ' num2str(k)], 'FontName', Font_Name, 'FontSize', Font_Size_Title);
    if k > 4
        xlabel('Time s', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
    end
    if mod(k, 4) == 1
        ylabel('Doppler Hz', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
    end
    set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
    ylim([-PRF/2 PRF/2]); % Limit Doppler range for better visibility
end

fprintf('Processing Complete.\n');