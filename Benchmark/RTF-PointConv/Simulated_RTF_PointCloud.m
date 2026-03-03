%% Script for RTF Point Cloud Generation with FPS for HAR
% Original Author: Hang Xu, Yong Li, Qingran Dong, Li Liu, Jingxia Li, Jianguo Zhang, and Bingjie Wang.
% Reproduced By: JoeyBG.
% Date: 2025-12-25.
% Affiliation: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description:
%   1. Simulates radar data for Gun Carrying and Walking activities.
%   2. Generates Range-Time-Frequency (RTF) 3D Cube via STFT stacking.
%   3. Extracts high-intensity points simulating CFAR detection.
%   4. Performs Farthest Point Sampling (FPS) to normalize point count.
%   5. Visualizes the 3D Point Cloud in Range-Time-Doppler space.

%% Initialization
clear all;
close all;
clc;
disp('---------- © Author: JoeyBG © ----------');

%% Parameter Definitions
% --- Activity Selection ---
activity_type = 'Walking';                                                  % Options: 'Walking' or 'GunCarrying'

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

% --- Point Cloud Processing Parameters ---
FPS_Point_Count = 1024;                                                     % Number of points after FPS
Intensity_Retention_Ratio = 0.15;                                           % Ratio of top intensity points to keep
STFT_Win_Size = 20;                                                         % Window size for STFT
STFT_Overlap = 18;                                                          % Overlap size for STFT

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
% --- 1.1 Scatterer Definition ---
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

% --- 1.2 Kinematic Trajectory Calculation ---
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

% --- 1.3 Echo Signal Generation ---
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

%% Section 2. RTF Cube Generation
fprintf('Constructing Range Time Frequency Cube...\n');

% 2.1 Range Profile and MTI
N_fft_range = 2^nextpow2(numADCSamples);
Range_Axis = (0:N_fft_range/2-1) * (fs / N_fft_range) * (c / (2 * K));

% FFT along fast time
Range_Profile_Complex = fft(Raw_Data, N_fft_range, 1);
Range_Profile_Complex = Range_Profile_Complex(1:N_fft_range/2, :);

% Wall Clutter Removal via MTI
RTM_Clutter_Removed = Range_Profile_Complex(:, 2:end) - Range_Profile_Complex(:, 1:end-1);
[Num_Range_Bins, Num_Pulses_MTI] = size(RTM_Clutter_Removed);

% 2.2 STFT Stacking to form RTF Cube
N_fft_doppler = 256;
STFT_Window = hamming(STFT_Win_Size);

% Temporary STFT to get dimensions
[~, f_vec, t_vec] = stft(zeros(1, Num_Pulses_MTI), PRF, 'Window', STFT_Window, 'OverlapLength', STFT_Overlap, 'FFTLength', N_fft_doppler);
Num_Time_Bins = length(t_vec);
Num_Freq_Bins = length(f_vec);

RTF_Cube = zeros(Num_Range_Bins, Num_Time_Bins, Num_Freq_Bins);

for r = 1:Num_Range_Bins
    time_series = RTM_Clutter_Removed(r, :);
    [S_stft, ~, ~] = stft(time_series, PRF, 'Window', STFT_Window, 'OverlapLength', STFT_Overlap, 'FFTLength', N_fft_doppler);
    RTF_Cube(r, :, :) = abs(S_stft).'; % Dimensions: Range x Time x Doppler
end

% Normalize RTF Cube
RTF_Cube = RTF_Cube / max(RTF_Cube(:));

%% Section 3. Point Cloud Generation and Sampling
fprintf('Generating Point Cloud with FPS...\n');

% 3.1 CFAR-Like Threshold Detection
threshold_val = quantile(RTF_Cube(:), 1 - Intensity_Retention_Ratio);
[r_idx, t_idx, d_idx] = ind2sub(size(RTF_Cube), find(RTF_Cube > threshold_val));
intensities = RTF_Cube(RTF_Cube > threshold_val);

% 3.2 Coordinate Mapping and Matrix Construction
P_range = Range_Axis(r_idx);
P_range = P_range(:); % Force column

P_time = t_vec(t_idx);
P_time = P_time(:); % Force column

P_doppler = f_vec(d_idx);
P_doppler = P_doppler(:); % Force column

% Construct N x 3 Matrix [Range, Time, Doppler]
Candidate_Points = [P_range, P_time, P_doppler];
Num_Candidates = size(Candidate_Points, 1);

fprintf('  Initial candidates after thresholding: %d\n', Num_Candidates);

% 3.3 FPS Processing
if Num_Candidates <= FPS_Point_Count
    Final_Points = Candidate_Points;
    Final_Intensities = intensities;
else
    sampled_indices = zeros(FPS_Point_Count, 1);
    dists = inf(Num_Candidates, 1);
    
    current_id = randi(Num_Candidates);
    
    for i = 1:FPS_Point_Count
        sampled_indices(i) = current_id;
        
        curr_pt = Candidate_Points(current_id, :);
        d_new = sum((Candidate_Points - curr_pt).^2, 2);
        
        dists = min(dists, d_new);
        [~, current_id] = max(dists);
    end
    
    Final_Points = Candidate_Points(sampled_indices, :);
    Final_Intensities = intensities(sampled_indices);
end

%% Section 4. Visualization
fprintf('Visualizing Results...\n');
figure('Name', ['RTF Point Cloud - ' activity_type], 'Color', 'w', 'Position', [100, 100, 1000, 600]);

% Create Scatter Plot (X: Time, Y: Doppler, Z: Range)
scatter3(Final_Points(:,2), Final_Points(:,3), Final_Points(:,1), ...
    20, Final_Intensities, 'filled');

% Apply Stylistics
colormap(JoeyBG_Colormap_Flip);
cb = colorbar;
cb.Label.String = 'Normalized Intensity';
cb.Label.FontName = Font_Name;
cb.Label.FontSize = Font_Size_Basis;

% Axis Labels and Limits
xlabel('Slow Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
ylabel('Doppler Frequency (Hz)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
zlabel('Range (m)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);

title(['RTF Point Cloud Sampling (FPS N=' num2str(FPS_Point_Count) ') - ' activity_type], ...
    'FontName', Font_Name, 'FontSize', Font_Size_Title);

set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
grid on;
box on;

% Adjust view
view(45, 30);
ylim([-PRF/2 PRF/2]);
zlim([0 6]);

fprintf('Processing Complete.\n');