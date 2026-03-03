%% Script for Complex-Valued 3D RTD Feature Generation for TWR HAR
% Original Author: Longzhen Tang, Shisheng Guo, Qiang Jian, Guolong Cui, Lingjiang Kong, and Xiaobo Yang.
% Reproduced By: JoeyBG.
% Date: 2025-12-27.
% Affiliation: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description:
%   1. Simulates radar data for Gun Carrying and Walking activities.
%   2. Implements Keystone Transform to correct Range Cell Migration.
%   3. Generates 3D Complex-Valued RTD Data via sliding window FFT.
%   4. Visualizes the Magnitude of the Complex 3D Feature in Range-Doppler-Time space.

%% Initialization
clear all;
close all;
clc;
disp('---------- © Author: JoeyBG © ----------');

%% Parameter Definitions
% --- Activity Selection ---
activity_type = 'Walking';                                                  % Activity selection

% --- Radar System Parameters ---
fc = 2.5e9;                                                                 % Radar carrier frequency (Hz)
c = 3e8;                                                                    % Speed of light (m/s)
lambda = c / fc;                                                            % Wavelength (m)
Tp = 40e-6;                                                                 % Pulse width (s)
B = 1e9;                                                                    % Chirp bandwidth (Hz)
K = B / Tp;                                                                 % Chirp rate (Hz/s)
PRF = 200;                                                                  % PRF (Hz)
fs = 4e6;                                                                   % ADC sampling rate (Hz)

% --- Simulation Time Parameters ---
sim_time = 1.0;                                                             % Simulation duration (s)
PRT = 1 / PRF;                                                              % Pulse Repetition Time
N_pulses = floor(sim_time * PRF);                                           % Total number of pulses
numADCSamples = floor(fs * Tp);                                             % Samples per chirp

% --- Wall Parameters ---
enable_wall = true;                                                         % Enable wall obstruction
wall_position_xyz = [1, 0, 1.25];                                           % Wall Center position
wall_dimensions_lwh = [0.24, 5, 2.5];                                       % Wall Dimensions
wall_epsilon_r = 6;                                                         % Dielectric constant
wall_loss_tangent = 0.03;                                                   % Loss tangent

% --- RTD Generation Parameters ---
Win_Length = 20;                                                            % Window length
Win_Stride = 2;                                                             % Stride
N_Doppler_FFT = 64;                                                         % Doppler FFT points
Intensity_Threshold_Ratio = 0.2;                                            % Visualization threshold

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
walking_angle_deg = 180;                                                    % Direction
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

% 1.2 Kinematic Trajectory Calculation
fprintf('Calculating kinematic trajectory for %s...\n', activity_type);
t_slow_axis = (0:N_pulses-1) * PRT;
t_fast_axis = (0:numADCSamples-1) / fs; 
pos = zeros(N_scatter, 3, N_pulses); 

walking_angle_rad = deg2rad(walking_angle_deg);
v_x = v_torso * cos(walking_angle_rad); 
v_y = v_torso * sin(walking_angle_rad); 

for m = 1:N_pulses
    t = t_slow_axis(m);
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

% 1.3 Echo Signal Generation
fprintf('Generating radar echo signals...\n');
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
        
        amplitude = sqrt(rcs(i)) / ((total_range/2)^2);
        
        if enable_wall
            if (tx_pos(1) < wall_x_back && scatter_pos(1) > wall_x_front) 
                    dist_in_wall = abs(wall_dimensions_lwh(1)); 
                    amplitude = amplitude * Transmission_Factor * exp(-alpha * dist_in_wall);
                    total_range = total_range + dist_in_wall*(sqrt(wall_epsilon_r)-1);
            end
        end
        
        % Dechirped Signal
        tau = total_range / c;
        beat_signal = amplitude .* exp(-1j * 2 * pi * (fc * tau + K * tau .* t_fast_axis));
        Raw_Data(:, m) = Raw_Data(:, m) + beat_signal';
    end
end

%% Section 2. Complex-Valued RTD Generation
% 2.1 Range Compression
fprintf('Performing Range Compression...\n');
N_fft_range = 2^nextpow2(numADCSamples);
Range_Profile = fft(Raw_Data, N_fft_range, 1);
Range_Profile = Range_Profile(1:N_fft_range/2, :); % Positive range bins only
[Num_Range_Bins, ~] = size(Range_Profile);

% MTI Clutter Suppression
Range_Profile_MTI = Range_Profile(:, 2:end) - Range_Profile(:, 1:end-1);
Range_Profile_MTI = [Range_Profile_MTI, Range_Profile_MTI(:,end)]; % Padding
[~, Num_Pulses_MTI] = size(Range_Profile_MTI);

% 2.2 Keystone Transform
fprintf('Applying Keystone Transform...\n');

RT_Keystone = zeros(size(Range_Profile_MTI));
slow_time_indices = 0:Num_Pulses_MTI-1;

% Frequency axis definition
freq_axis_beat = (0:Num_Range_Bins-1)' * (fs / N_fft_range);
freq_axis_rf = fc + freq_axis_beat; 

for r = 1:Num_Range_Bins
    % Scaling factor
    f_curr = freq_axis_rf(r);
    scale_factor = f_curr / fc;
    
    % Linear interpolation
    t_new = slow_time_indices * scale_factor;
    RT_Keystone(r, :) = interp1(slow_time_indices, Range_Profile_MTI(r, :), t_new, 'linear', 0);
end

% 2.3 3D Complex-Valued RTD Construction
fprintf('Constructing 3D Complex RTD Cube...\n');

num_windows = floor((Num_Pulses_MTI - Win_Length) / Win_Stride) + 1;
RTD_Complex_Cube = zeros(Num_Range_Bins, N_Doppler_FFT, num_windows);

Doppler_Window = hamming(Win_Length);

for w = 1:num_windows
    idx_start = (w-1)*Win_Stride + 1;
    idx_end = idx_start + Win_Length - 1;
    
    % Segment extraction
    segment = RT_Keystone(:, idx_start:idx_end);
    
    % Windowing
    segment_win = segment .* Doppler_Window';
    
    % Slow time FFT to Doppler
    RD_Map = fftshift(fft(segment_win, N_Doppler_FFT, 2), 2);
    
    RTD_Complex_Cube(:, :, w) = RD_Map;
end

%% Section 3. Visualization
fprintf('Visualizing Complex-Valued RTD Feature...\n');

% Magnitude calculation
RTD_Magnitude = abs(RTD_Complex_Cube);
RTD_Magnitude = RTD_Magnitude / max(RTD_Magnitude(:)); 

% Axes setup
Range_Axis = (0:Num_Range_Bins-1) * (c / (2 * B));
Doppler_Axis = linspace(-PRF/2, PRF/2, N_Doppler_FFT);
Time_Axis = (0:num_windows-1) * Win_Stride * PRT;

% Thresholding for 3D Scatter Visualization
threshold_val = quantile(RTD_Magnitude(:), 1 - Intensity_Threshold_Ratio);
[r_idx, d_idx, t_idx] = ind2sub(size(RTD_Magnitude), find(RTD_Magnitude > threshold_val));
intensities = RTD_Magnitude(RTD_Magnitude > threshold_val);

% Index conversion
P_range = Range_Axis(r_idx)';
P_doppler = Doppler_Axis(d_idx)';
P_time = Time_Axis(t_idx)';

figure('Name', ['Complex RTD Feature - ' activity_type], 'Color', 'w', 'Position', [100, 100, 1000, 600]);

% 3D Scatter Plot
scatter3(P_time, P_doppler, P_range, 20, intensities, 'filled', ...
    'MarkerFaceAlpha', 0.6);

% Stylistics
colormap(JoeyBG_Colormap_Flip);
cb = colorbar;
cb.Label.String = '|Complex RTD| Normalized';
cb.Label.FontName = Font_Name;
cb.Label.FontSize = Font_Size_Basis;

% Axis Labels
xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
ylabel('Doppler (Hz)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
zlabel('Range (m)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
title(['3D Complex-Valued RTD Feature - ' activity_type], 'FontName', Font_Name, 'FontSize', Font_Size_Title);

set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
grid on;
box on;

% Limits
ylim([-100 100]); 
zlim([0 6]);   

view(45, 30);

% Output Summary
fprintf('Processing Complete.\n');
fprintf('Resulting 3D Feature Size: [%d Range x %d Doppler x %d Time]\n', ...
    Num_Range_Bins, N_Doppler_FFT, num_windows);