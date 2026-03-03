%% Script for Simulated RTD Feature Dataset Generator
% Original Author: Longzhen Tang, Shisheng Guo, Qiang Jian, Guolong Cui, Lingjiang Kong, and Xiaobo Yang.
% Reproduced By: JoeyBG.
% Date: 2025-12-27.
% Affiliation: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description: 
%   1. Generates synthetic 3D Complex-Valued RTD datasets for HAR.
%   2. Follows the class structure (P1-P4, Gun/NoGun) of SimH dataset.
%   3. Processes: Echo Sim -> Range Compression -> MTI -> Keystone Transform -> Sliding Window FFT.
%   4. Outputs .mat files containing 'RTD_Feature'.
%   5. Simulation time set to 1s.

%% Initialization
clear all;
close all;
clc;
disp('---------- © Author: JoeyBG © ----------');

%% Global Parameter Definitions
% --- Radar System Parameters ---
fc = 2.5e9;                                                                 % Carrier frequency (Hz)
c = 3e8;                                                                    % Speed of light (m/s)
lambda = c / fc;                                                            % Wavelength (m)
Tp = 40e-6;                                                                 % Pulse width (s)
B = 1e9;                                                                    % Bandwidth (Hz)
K = B / Tp;                                                                 % Chirp rate (Hz/s)
PRF = 200;                                                                  % PRF (Hz)
fs = 4e6;                                                                   % Sampling rate (Hz)

% --- Simulation Time Parameters ---
sim_time = 1;                                                               % It can be adjusted to 1.44s to match paper's ConvGRU input
PRT = 1 / PRF;                                      
N_pulses = floor(sim_time * PRF);                   
numADCSamples = floor(fs * Tp);                     

% --- Wall Parameters ---
enable_wall = true;                                 
wall_position_xyz_base = [1, 0, 1.25];                   
wall_dimensions_lwh = [0.24, 5, 2.5];               
wall_epsilon_r = 6;                                 
wall_loss_tangent = 0.03;                           

% --- RTD Feature Generation Parameters ---
Win_Length = 20;                                                            % Window length (pulses)
Win_Stride = 2;                                                             % Window stride (pulses)
N_Doppler_FFT = 64;                                                         % Doppler FFT size

% --- Antenna Configuration ---
tx_pos = [0, 0, 1.5];
rx_pos = [0, 0.1, 1.5];

%% Dataset Configuration
% Define Output Path
root_RTDSet = 'Simulated_RTDSet';

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
create_dirs(root_RTDSet, classes);

%% Main Generation Loop
total_samples = sum([classes.Count]);
current_sample = 0;

for cls_idx = 1:length(classes)
    cls = classes(cls_idx);
    fprintf('\nProcessing Class: %s (Target: %d samples)\n', cls.Name, cls.Count);
    
    for grp_idx = 1:cls.Count
        current_sample = current_sample + 1;
        fprintf('  [Total Progress: %.1f%%] Generating Sample %d/%d for %s...\n', ...
            (current_sample/total_samples)*100, grp_idx, cls.Count, cls.Name);
        
        %% 1. Parameter Randomization
        % Vary initial position (+- 0.5m)
        rand_pos_x = 2.5 + (rand - 0.5) * 0.5; 
        rand_pos_y = 0 + (rand - 0.5) * 0.5;
        initial_position_xy = [rand_pos_x, rand_pos_y];
        
        % Vary walking angle (+- 15 degrees) around 180
        base_angle = 180; 
        walking_angle_deg = base_angle + (rand - 0.5) * 30;
        
        % Vary velocity slightly (+- 5%)
        v_torso = cls.Velocity * (1 + (rand - 0.5) * 0.1);
        
        % Vary gait frequency slightly based on velocity
        f_gait = (v_torso / cls.Velocity) * 1.0; 
        
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
        
        % Kinematic Trajectory Calculation
        t_slow = (0:N_pulses-1) * PRT;
        t_fast_axis = (0:numADCSamples-1) / fs; 
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

        %% 3. Echo Signal Generation
        % Wall constants
        eta0 = 376.73; eta_wall = eta0 / sqrt(wall_epsilon_r);
        Transmission_Factor = (2*eta_wall/(eta_wall+eta0)) * (2*eta0/(eta_wall+eta0));
        alpha_wall = pi * fc / c * sqrt(wall_epsilon_r) * wall_loss_tangent;
        wall_x_front = wall_position_xyz_base(1) + wall_dimensions_lwh(1)/2;
        wall_x_back = wall_position_xyz_base(1) - wall_dimensions_lwh(1)/2;
        
        Raw_Data = zeros(numADCSamples, N_pulses);
        
        for m = 1:N_pulses
            for i = 1:N_scatter
                scatter_pos = squeeze(pos(i, :, m));
                dist_tx = norm(scatter_pos - tx_pos);
                dist_rx = norm(scatter_pos - rx_pos);
                total_range = dist_tx + dist_rx;
                
                amplitude = sqrt(rcs(i)) / ((total_range/2)^2);
                
                % Wall Effect
                if enable_wall
                    if (tx_pos(1) < wall_x_back && scatter_pos(1) > wall_x_front) 
                            dist_in_wall = abs(wall_dimensions_lwh(1)); 
                            amplitude = amplitude * Transmission_Factor * exp(-alpha_wall * dist_in_wall);
                            total_range = total_range + dist_in_wall*(sqrt(wall_epsilon_r)-1);
                    end
                end
                
                tau = total_range / c;
                beat_signal = amplitude .* exp(-1j * 2 * pi * (fc * tau + K * tau .* t_fast_axis));
                Raw_Data(:, m) = Raw_Data(:, m) + beat_signal';
            end
        end

        %% 4. Complex RTD Feature Construction
        % 4.1 Range Compression
        N_fft_range = 2^nextpow2(numADCSamples);
        Range_Profile = fft(Raw_Data, N_fft_range, 1);
        Range_Profile = Range_Profile(1:N_fft_range/2, :); 
        [Num_Range_Bins, ~] = size(Range_Profile);

        % 4.2 MTI
        Range_Profile_MTI = Range_Profile(:, 2:end) - Range_Profile(:, 1:end-1);
        Range_Profile_MTI = [Range_Profile_MTI, Range_Profile_MTI(:,end)]; 
        [~, Num_Pulses_MTI] = size(Range_Profile_MTI);

        % 4.3 Keystone Transform
        RT_Keystone = zeros(size(Range_Profile_MTI));
        slow_time_indices = 0:Num_Pulses_MTI-1;
        freq_axis_beat = (0:Num_Range_Bins-1)' * (fs / N_fft_range);
        freq_axis_rf = fc + freq_axis_beat; 

        for r = 1:Num_Range_Bins
            % Resampling factor
            f_curr = freq_axis_rf(r);
            scale_factor = f_curr / fc;
            
            % Linear interpolation along slow time
            t_new = slow_time_indices * scale_factor;
            RT_Keystone(r, :) = interp1(slow_time_indices, Range_Profile_MTI(r, :), t_new, 'linear', 0);
        end

        % 4.4 Sliding Window 3D Cube Construction
        num_windows = floor((Num_Pulses_MTI - Win_Length) / Win_Stride) + 1;
        RTD_Feature = zeros(Num_Range_Bins, N_Doppler_FFT, num_windows);
        
        Doppler_Window = hamming(Win_Length);

        for w = 1:num_windows
            idx_start = (w-1)*Win_Stride + 1;
            idx_end = idx_start + Win_Length - 1;
            
            segment = RT_Keystone(:, idx_start:idx_end);
            segment_win = segment .* Doppler_Window';
            
            % FFT along slow time -> Doppler
            RD_Map = fftshift(fft(segment_win, N_Doppler_FFT, 2), 2);
            
            RTD_Feature(:, :, w) = RD_Map;
        end
        % RTD_Feature is Complex Double: [Range x Doppler x Time]

        %% 6. Save Data        
        filename = sprintf('%s_Sample_%d.mat', cls.Name, grp_idx);
        save_path = fullfile(root_RTDSet, cls.Name, filename);
        
        save(save_path, 'RTD_Feature');
        
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