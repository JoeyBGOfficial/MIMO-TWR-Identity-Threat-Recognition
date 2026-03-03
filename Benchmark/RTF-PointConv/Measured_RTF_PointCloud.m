%% Script for Measured RTF Point Cloud Generation with FPS
% Original Author: Hang Xu, Yong Li, Qingran Dong, Li Liu, Jingxia Li, Jianguo Zhang, and Bingjie Wang.
% Improved By: JoeyBG.
% Date: 2025-12-25.
% Affiliate: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description:
%   1 Reads raw measured radar data .NOV using Read_NOV function
%   2 Preprocesses data including Wall Compensation MTI and Entropy based Channel Selection
%   3 Generates Range Time Frequency RTF 3D Cube via STFT stacking
%   4 Extracts high intensity points simulating CFAR detection
%   5 Performs Farthest Point Sampling FPS to normalize point count
%   6 Visualizes the 3D Point Cloud in Range Time Doppler space

%% Initialization
clear all;
close all;
clc;
disp('---------- © Author: JoeyBG © ----------');

% Load configurations
try
    load("JoeyBG_Colormap.mat");                                            % Load custom colormap
    load("Array_Configurations\data_info.mat");                             % Load radar config
catch
    warning('Configuration files missed. Using defaults.');
    if ~exist('data_info','var')
        data_info.f0 = 2.5e9; 
        data_info.B = 1.0e9; 
        data_info.PRT = 1e-3; 
        data_info.fs = 4e6; 
    end
end

%% Parameter Definitions
% --- File Parameters ---
FilePath = "D:\JoeyBG_Research_Production\TWR_Identity_Threat\RW_Set\RW_Datas\P1_Gun.NOV";
Total_Readin_Packages = 100;                                                % Max packages
Fast_Time_Points = 3940;      
Readin_DS_Ratio = 1;                                                        % Reading DS ratio
Process_DS_Ratio = 5;                                                       % Processing DS ratio
Total_DS_Ratio = Readin_DS_Ratio * Process_DS_Ratio;

% --- Radar System Parameters ---
c = 3e8;                                                                    % Speed of light
fc = data_info.f0;
B = data_info.B;              
T = data_info.PRT;            
fs = data_info.fs;            
PRF_Raw = 1/T;                
PRF_Effective = PRF_Raw / Total_DS_Ratio;                                   % Effective PRF

% --- Wall Parameters ---
Thickness_Wall = 0.12;    
Wall_Dieletric = 6;       
Wall_Compensation_Dist = Thickness_Wall * (sqrt(Wall_Dieletric) - 1);       

% --- Range Axis Calculation ---
FFT_Points = Fast_Time_Points;          
Range_Resolution = c / (2 * B);
Range_Axis_Raw = ((0:FFT_Points/2-1) * Range_Resolution) / 2;               
Range_Axis_Physical = Range_Axis_Raw - Wall_Compensation_Dist;

% --- ROI Selection ---
Min_Range_ROI = 0.5;                                                          
Max_Range_ROI = 6.0;                                                                                                                                  
[~, Min_Bin_Idx] = min(abs(Range_Axis_Physical - Min_Range_ROI));
[~, Max_Bin_Idx] = min(abs(Range_Axis_Physical - Max_Range_ROI));
Range_Axis_Show = Range_Axis_Physical(Min_Bin_Idx:Max_Bin_Idx);             
MTI_Step = 1;

% --- Point Cloud Processing Parameters ---
FPS_Point_Count = 1024;                                                     % Target number of points
Intensity_Retention_Ratio = 0.05;                                           % Top intensity ratio
STFT_Win_Size = 20;                                                         % Window size for STFT
STFT_Overlap = 18;                                                          % Overlap size
N_fft_doppler = 256;                                                        % Doppler FFT size

% --- Visualization Parameters ---
Font_Name = 'Palatino Linotype'; 
Font_Size_Basis = 12;
Font_Size_Title = 14;
if ~exist('JoeyBG_Colormap', 'var')
    JoeyBG_Colormap = jet(256);
end

%% Section 1 Data Reading and Preprocessing
fprintf('Reading data file: %s ...\n', FilePath);

% 1.1 Read Data
% Returns [FastTime, SlowTime, Channels] after internal reshape in Read_NOV logic
[adc_data_raw, ~] = Read_NOV(FilePath, Total_Readin_Packages, 8, 8, Fast_Time_Points, Readin_DS_Ratio);
% Permute to [FastTime, Channels, SlowTime] for consistency with processing script
adc_data = permute(adc_data_raw, [1 3 2]); 
[~, Total_Channels, Total_Frames_Read] = size(adc_data);

% 1.2 RTM Generation and MTI
fprintf('Generating RTM and performing MTI...\n');
% Range Compression
Range_Profile_Complex = fft(adc_data, FFT_Points, 1);
% Cut ROI
RTM_ROI_Raw = Range_Profile_Complex(Min_Bin_Idx:Max_Bin_Idx, :, :);
% MTI Filter
RTM_MTI = RTM_ROI_Raw(:, :, MTI_Step+1:end) - RTM_ROI_Raw(:, :, 1:end-MTI_Step);
[nRange, ~, Final_Frames] = size(RTM_MTI);

% 1.3 Channel Selection
fprintf('Selecting best channel via Entropy...\n');
RTM_Mag_Stack = zeros(nRange, Final_Frames, Total_Channels);
entropies = zeros(1, Total_Channels);

for ch = 1:Total_Channels
    temp = abs(squeeze(RTM_MTI(:, ch, :)));
    RTM_Mag_Stack(:,:,ch) = mat2gray(temp);
    entropies(ch) = entropy(im2uint8(RTM_Mag_Stack(:,:,ch)));
end

[min_entropy_val, Ref_Idx] = min(entropies);
fprintf('  > Selected Channel: %d (Entropy: %.4f)\n', Ref_Idx, min_entropy_val);

% Extract the best channel data for RTF generation
Best_RTM_Complex = squeeze(RTM_MTI(:, Ref_Idx, :)); % Dimensions: [Range, Time]

%% Section 2 RTF Cube Generation
fprintf('Constructing Range Time Frequency Cube...\n');

STFT_Window = hamming(STFT_Win_Size);

% Temporary STFT to get dimensions
[~, f_vec, t_vec] = stft(zeros(1, Final_Frames), PRF_Effective, ...
    'Window', STFT_Window, 'OverlapLength', STFT_Overlap, 'FFTLength', N_fft_doppler);
Num_Time_Bins = length(t_vec);
Num_Freq_Bins = length(f_vec);

RTF_Cube = zeros(nRange, Num_Time_Bins, Num_Freq_Bins);

% Calculate STFT for each range bin
for r = 1:nRange
    time_series = Best_RTM_Complex(r, :);
    [S_stft, ~, ~] = stft(time_series, PRF_Effective, ...
        'Window', STFT_Window, 'OverlapLength', STFT_Overlap, 'FFTLength', N_fft_doppler);
    RTF_Cube(r, :, :) = abs(S_stft).'; % Dimensions Range x Time x Doppler
end

% Normalize RTF Cube
RTF_Cube = RTF_Cube / max(RTF_Cube(:));

%% Section 3 Point Cloud Generation and Sampling
fprintf('Generating Point Cloud with FPS...\n');

% 3.1 CFAR-Like Thresholding
threshold_val = quantile(RTF_Cube(:), 1 - Intensity_Retention_Ratio);
[r_idx, t_idx, d_idx] = ind2sub(size(RTF_Cube), find(RTF_Cube > threshold_val));
intensities = RTF_Cube(RTF_Cube > threshold_val);

% 3.2 Coordinate Mapping
P_range = Range_Axis_Show(r_idx);
P_range = P_range(:); % Force column

P_time = t_vec(t_idx);
P_time = P_time(:);   % Force column

P_doppler = f_vec(d_idx);
P_doppler = P_doppler(:); % Force column

% Construct N x 3 Matrix [Range, Time, Doppler]
Candidate_Points = [P_range, P_time, P_doppler];
Num_Candidates = size(Candidate_Points, 1);

fprintf('  Initial candidates after thresholding: %d\n', Num_Candidates);

% 3.3 FPS Processing
if Num_Candidates <= FPS_Point_Count
    % If fewer points than target, use all and pad if necessary
    Final_Points = Candidate_Points;
    Final_Intensities = intensities;
    if Num_Candidates < FPS_Point_Count
        warning('Candidate points fewer than target FPS count.');
    end
else
    sampled_indices = zeros(FPS_Point_Count, 1);
    dists = inf(Num_Candidates, 1);
    
    % Random start
    current_id = randi(Num_Candidates);
    
    % Data normalization for distance calculation
    C_min = min(Candidate_Points);
    C_max = max(Candidate_Points);
    C_Norm = (Candidate_Points - C_min) ./ (C_max - C_min + 1e-6);
    
    for i = 1:FPS_Point_Count
        sampled_indices(i) = current_id;
        
        curr_pt_norm = C_Norm(current_id, :);
        % Calculate squared Euclidean distance
        d_new = sum((C_Norm - curr_pt_norm).^2, 2);
        
        dists = min(dists, d_new);
        [~, current_id] = max(dists);
    end
    
    Final_Points = Candidate_Points(sampled_indices, :);
    Final_Intensities = intensities(sampled_indices);
end

%% Section 4 Visualization
fprintf('Visualizing Results...\n');
figure('Name', 'Measured RTF Point Cloud', 'Color', 'w', 'Position', [100, 100, 1000, 600]);

% Create Scatter Plot (X: Time, Y: Doppler, Z: Range)
scatter3(Final_Points(:,2), Final_Points(:,3), Final_Points(:,1), ...
    20, Final_Intensities, 'filled');

% Apply Stylistics
colormap(JoeyBG_Colormap);
cb = colorbar;
cb.Label.String = 'Normalized Intensity';
cb.Label.FontName = Font_Name;
cb.Label.FontSize = Font_Size_Basis;

% Axis Labels and Limits
xlabel('Slow Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
ylabel('Doppler Frequency (Hz)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
zlabel('Range (m)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);

title(['Measured RTF Point Cloud (FPS N=' num2str(FPS_Point_Count) ')'], ...
    'FontName', Font_Name, 'FontSize', Font_Size_Title);

set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
grid on;
box on;

% Adjust view
view(45, 30);
ylim([-PRF_Effective/2 PRF_Effective/2]);
zlim([Min_Range_ROI Max_Range_ROI]);

fprintf('Processing Complete.\n');