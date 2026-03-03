%% Script for DTM Dataset Generation with Multistatic MIMO Bio Radar
% Original Author: Yimeng Zhao, Yong Jia, Dong Huang, Li Zhang, Yao Zheng, Jianqi Wang, and Fugui Qi.
% Reproduced By: JoeyBG.
% Date: 2025-12-28.
% Affiliation: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description:
%   1 Simulates radar data for Walking and GunCarrying activities using a MIMO array.
%   2 Array configuration includes 2 Tx and 4 Rx antennas yielding 8 virtual channels.
%   3 Generates synthetic datasets for 4 subjects with different physical attributes.
%   4 Performs Range Doppler processing to extract DTMs for each channel.
%   5 Saves normalized DTM spectrograms into channel-specific directory structures.

%% Initialization
clear all;
close all;
clc;
disp('---------- © Author: JoeyBG © ----------');

%% Parameter Definitions
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
wall_position_xyz_base = [1, 0, 1.25];                                      % Wall Center position
wall_dimensions_lwh = [0.24, 5, 2.5];                                       % Wall Dimensions
wall_epsilon_r = 6;                                                         % Dielectric constant
wall_loss_tangent = 0.03;                                                   % Loss tangent

% --- Spectrogram Parameters ---
STFT_Win_Size = 20;                                                         % Window size for STFT
STFT_Overlap = 18;                                                          % Overlap size for STFT
N_fft_doppler = 256;                                                        % Doppler FFT size
STFT_Window = hamming(STFT_Win_Size);

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

%% Dataset Configuration
% Root directories for each channel will be created dynamically
root_base_name = 'Simulated_DTMSet_Channel';

% Define Classes with varying heights and velocities
classes = struct(...
    'Name', {'P1_Gun', 'P1_Nogun', 'P2_Gun', 'P2_Nogun', ...
             'P3_Gun', 'P3_Nogun', 'P4_Gun', 'P4_Nogun'}, ...
    'Height', {1.8, 1.8, 1.7, 1.7, 1.6, 1.6, 1.5, 1.5}, ...
    'Velocity', {1.2, 1.2, 1.13, 1.13, 1.07, 1.07, 1.0, 1.0}, ...
    'Activity', {'GunCarrying', 'Walking', 'GunCarrying', 'Walking', ...
                 'GunCarrying', 'Walking', 'GunCarrying', 'Walking'}, ...
    'Count', {368, 296, 242, 118, 118, 118, 118, 118} ...                   % Adjusted count for demonstration
);

% Initialize Directory Structure
for ch = 1:num_channels
    root_path = sprintf('%s%d', root_base_name, ch);
    create_dirs(root_path, classes);
end

%% Main Generation Loop
total_samples = sum([classes.Count]);
global_counter = 0;

for cls_idx = 1:length(classes)
    cls = classes(cls_idx);
    fprintf('\nProcessing Class: %s (Target: %d samples)\n', cls.Name, cls.Count);
    
    for grp_idx = 1:cls.Count
        global_counter = global_counter + 1;
        fprintf('  [Progress: %.2f%%] Generating Sample %d/%d for %s...\n', ...
            (global_counter/total_samples)*100, grp_idx, cls.Count, cls.Name);
        
        %% 1. Parameter Randomization
        % Vary initial position slightly
        rand_pos_x = 2.5 + (rand - 0.5) * 0.5; 
        rand_pos_y = 0 + (rand - 0.5) * 0.5;
        initial_position_xy = [rand_pos_x, rand_pos_y];
        
        % Vary walking angle slightly around 180 degrees
        base_angle = 180; 
        walking_angle_deg = base_angle + (rand - 0.5) * 20;
        
        % Vary velocity slightly
        v_torso = cls.Velocity * (1 + (rand - 0.5) * 0.1);
        
        % Vary gait frequency based on velocity
        f_gait = (v_torso / cls.Velocity) * 1.2; 
        
        activity_type = cls.Activity;
        person_height = cls.Height;

        %% 2. Human Kinematic Model Setup
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
        A_thigh = deg2rad(30);                              
        A_calf = deg2rad(45);                               
        A_arm = deg2rad(35);                                
        
        % Scatterer Definition
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

        % Kinematic Trajectory Calculation
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

        %% 3. MIMO Echo Signal Generation
        % Wall constants calculation
        if enable_wall
            eta0 = 376.73; eta_wall = eta0 / sqrt(wall_epsilon_r);
            Transmission_Factor = (2*eta_wall/(eta_wall+eta0)) * (2*eta0/(eta_wall+eta0));
            alpha = pi * fc / c * sqrt(wall_epsilon_r) * wall_loss_tangent;
            wall_x_front = wall_position_xyz_base(1) + wall_dimensions_lwh(1)/2;
            wall_x_back = wall_position_xyz_base(1) - wall_dimensions_lwh(1)/2;
        end

        % Data container FastTime x SlowTime x Channel
        Raw_Data_MIMO = zeros(numADCSamples, N_pulses, num_channels);

        for k = 1:num_channels
            tx_idx = channels(k, 1);
            rx_idx = channels(k, 2);
            
            curr_tx_pos = tx_locs(tx_idx, :);
            curr_rx_pos = rx_locs(rx_idx, :);
            
            for m = 1:N_pulses
                for i = 1:N_scatter
                    scatter_pos = squeeze(pos(i, :, m));
                    dist_tx = norm(scatter_pos - curr_tx_pos);
                    dist_rx = norm(scatter_pos - curr_rx_pos);
                    total_range = dist_tx + dist_rx;
                    
                    amplitude = sqrt(rcs(i)) / ((total_range/2)^2);
                    
                    if enable_wall
                        if (curr_tx_pos(1) < wall_x_back && scatter_pos(1) > wall_x_front) 
                                dist_in_wall = abs(wall_dimensions_lwh(1)); 
                                amplitude = amplitude * Transmission_Factor * exp(-alpha * dist_in_wall);
                                total_range = total_range + dist_in_wall*(sqrt(wall_epsilon_r)-1);
                        end
                    end
                    
                    tau = total_range / c;
                    beat_signal = amplitude .* exp(-1j * 2 * pi * (fc * tau + K * tau .* t_fast));
                    Raw_Data_MIMO(:, m, k) = Raw_Data_MIMO(:, m, k) + beat_signal';
                end
            end
        end

        %% 4. DTM Processing and Saving
        filename = sprintf('%s_Sample_%d.mat', cls.Name, grp_idx);
        
        for k = 1:num_channels
            % 4.1 Range Profile and MTI per channel
            raw_k = Raw_Data_MIMO(:, :, k);
            N_fft_range = 2^nextpow2(numADCSamples);
            
            % FFT along fast time
            Range_Profile_Complex = fft(raw_k, N_fft_range, 1);
            Range_Profile_Complex = Range_Profile_Complex(1:N_fft_range/2, :);
            
            % Wall Clutter Removal via MTI
            RTM_Clutter_Removed = Range_Profile_Complex(:, 2:end) - Range_Profile_Complex(:, 1:end-1);
            
            % 4.2 Sum along range bins to get time series for DTM
            dtm_time_series = sum(RTM_Clutter_Removed, 1);
            
            % 4.3 STFT
            [S_stft, ~, ~] = stft(dtm_time_series, PRF, 'Window', STFT_Window, 'OverlapLength', STFT_Overlap, 'FFTLength', N_fft_doppler);
            
            % Normalize Amplitude
            Mag_DTM = abs(S_stft);
            Norm_DTM = Mag_DTM / max(Mag_DTM(:));
            
            % 4.4 Save to corresponding channel folder
            root_path = sprintf('%s%d', root_base_name, k);
            save_dir = fullfile(root_path, cls.Name);
            save_full_path = fullfile(save_dir, filename);
            
            % Save the normalized DTM matrix
            save(save_full_path, 'Norm_DTM');
        end
    end
end
disp('Dataset Generation Complete.');

%% Helper Functions
function create_dirs(root, classes)
    if ~exist(root, 'dir'), mkdir(root); end
    for i = 1:length(classes)
        subpath = fullfile(root, classes(i).Name);
        if ~exist(subpath, 'dir'), mkdir(subpath); end
    end
end