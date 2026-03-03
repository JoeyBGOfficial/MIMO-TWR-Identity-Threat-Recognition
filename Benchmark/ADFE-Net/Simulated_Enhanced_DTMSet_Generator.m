%% Script for Simulated Enhanced DTM Dataset Generator
% Original Author: Longzhen Tang, Shisheng Guo, Jiachen Li, Junda Zhu, Guolong Cui, Lingjiang Kong, and Xiaobo Yang.
% Reproduced By: JoeyBG.
% Date: 2025-12-29.
% Affiliation: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description: 
%   1. Generates synthetic Enhanced Doppler Spectrograms using ADFE fusion.
%   2. Follows the class structure (P1-P4, Gun/NoGun) of SimH dataset.
%   3. Processes: Echo Sim -> Range Profile -> TPC/AC Suppression -> STFT -> Weighted Fusion.
%   4. Outputs .mat files containing 'Enhanced_DTM' matrices.
%   5. Saved to path: 'Simulated_Enhanced_DTMSet/'.

%% Initialization
clear all;
close all;
clc;
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
sim_time = 1.0;                                                             % Duration (s)
PRT = 1 / PRF;                                      
N_pulses = floor(sim_time * PRF);                   
numADCSamples = floor(fs * Tp);                     

% --- Wall Parameters ---
enable_wall = true;                                 
wall_position_xyz_base = [1, 0, 1.25];                   
wall_dimensions_lwh = [0.24, 5, 2.5];               
wall_epsilon_r = 6;                                 
wall_loss_tangent = 0.03;                           

% --- Spectrogram & Fusion Parameters ---
STFT_Win_Size = 20;                                                         % Window size for STFT
STFT_Overlap = 18;                                                          % Overlap size for STFT
N_fft_doppler = 512;                                                        % Doppler FFT size
Fusion_Bandwidth_Sigma = 10;                                                % Sigma for Gaussian fusion mask

% --- Antenna Configuration ---
tx_pos = [0, 0, 1.5];
rx_pos = [0, 0.1, 1.5];

%% Dataset Configuration
% Define Output Path
root_DTMSet = 'Simulated_Enhanced_DTMSet';

% Define Classes (4 Persons x 2 Activities)
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
create_dirs(root_DTMSet, classes);

%% Main Generation Loop
total_samples = sum([classes.Count]);
current_sample = 0;

% Pre-define STFT Window to save time
STFT_Window = hamming(STFT_Win_Size);
% Fusion Weight Mask Calculation (Fixed based on FFT size)
freq_indices = linspace(-PRF/2, PRF/2, N_fft_doppler)';
Weight_AC_Vec = exp(- (freq_indices.^2) / (2 * Fusion_Bandwidth_Sigma^2));
Weight_TPC_Vec = 1 - Weight_AC_Vec;

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

        %% 3. Echo Signal Generation
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
                beat_signal = amplitude .* exp(-1j * 2 * pi * (fc * tau + K * tau .* t_fast));
                Raw_Data(:, m) = Raw_Data(:, m) + beat_signal';
            end
        end

        %% 4. Range Profile & Clutter Suppression
        N_fft_range = 2^nextpow2(numADCSamples);
        Range_Profile_Complex = fft(Raw_Data, N_fft_range, 1);
        Range_Profile_Complex = Range_Profile_Complex(1:N_fft_range/2, :);

        % 4.1 Two-Pulse Cancellation (TPC)
        RTM_TPC = Range_Profile_Complex(:, 2:end) - Range_Profile_Complex(:, 1:end-1);

        % 4.2 Average Cancellation (AC)
        Mean_Clutter = mean(Range_Profile_Complex, 2);
        RTM_AC = Range_Profile_Complex - Mean_Clutter;
        RTM_AC = RTM_AC(:, 1:end-1); % Align with TPC dimensions

        %% 5. Enhanced Spectrogram Generation
        % Function: Input [Range x Time], Transpose to [Time x Range] for STFT, 
        % Output [Freq x Time x Range], Sum over Range dim -> [Freq x Time]
        generate_DTM = @(RTM_Input) squeeze(sum(abs(stft(RTM_Input.', PRF, ...
            'Window', STFT_Window, 'OverlapLength', STFT_Overlap, 'FFTLength', N_fft_doppler)), 3));

        % Generate Basic Spectrograms
        DTM_TPC = generate_DTM(RTM_TPC);
        DTM_AC  = generate_DTM(RTM_AC);
        
        % Normalize (0-1)
        max_tpc = max(DTM_TPC(:)); if max_tpc == 0, max_tpc = 1; end
        max_ac  = max(DTM_AC(:));  if max_ac == 0, max_ac = 1; end
        DTM_TPC_Norm = DTM_TPC / max_tpc;
        DTM_AC_Norm  = DTM_AC / max_ac;

        % Expand Frequency Weights to Map (Doppler x Time)
        % Note: t_vec length is determined by STFT output
        Num_Time_Bins_Final = size(DTM_TPC_Norm, 2);
        Weight_AC_Map = repmat(Weight_AC_Vec, 1, Num_Time_Bins_Final);
        Weight_TPC_Map = repmat(Weight_TPC_Vec, 1, Num_Time_Bins_Final);

        % Fusion
        Enhanced_DTM = (Weight_TPC_Map .* DTM_TPC_Norm) + (Weight_AC_Map .* DTM_AC_Norm);
        Enhanced_DTM = Enhanced_DTM / max(Enhanced_DTM(:)); % Re-normalize final result

        %% 6. Save Data        
        filename = sprintf('%s_Enhanced_%d.mat', cls.Name, grp_idx);
        save_path = fullfile(root_DTMSet, cls.Name, filename);
        
        % Save variable 'Enhanced_DTM' [Freq x Time]
        save(save_path, 'Enhanced_DTM');
        
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