%% Script for Measured Complex-Valued RTD Feature Generation
% Original Author: Longzhen Tang, Shisheng Guo, Qiang Jian, Guolong Cui, Lingjiang Kong, and Xiaobo Yang.
% Reproduced By: JoeyBG.
% Date: 2025-12-27.
% Affiliation: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description:
%   1. Reads raw binary radar data (.NOV) using strict parameter definitions from Measured_RTMs.m.
%   2. Processes a specific MIMO channel.
%   3. Implements the Paper's pipeline: Range Comp -> MTI -> Keystone -> Sliding Window.
%   4. Visualizes the Measured 3D Complex-Valued RTD Feature.

%% Initialization
clear all;
close all;
clc;
disp('---------- © Author: JoeyBG © ----------');

% Load configurations
try
    load("JoeyBG_Colormap.mat");                                            % Custom colormap
    if exist('CList_Flip', 'var')
        MyColormap = CList_Flip;
    else
        MyColormap = jet;
    end
    load("Array_Configurations\data_info.mat");                             % Radar configuration
catch
    warning('Configuration files missed. Using defaults and Jet colormap.');
    MyColormap = jet;
    if ~exist('data_info','var')
        data_info.f0 = 2.5e9; 
        data_info.B = 1.0e9; 
        data_info.PRT = 1e-3; 
        data_info.fs = 4e6; 
    end
end

%% Parameter Definitions
% --- Physical Constants ---
c = 3e8;                                                                    % Speed of light (m/s)

% --- Wall Parameters ---
Thickness_Wall = 0.12;                                                      % Wall thickness (m)
Wall_Dieletric = 6;                                                         % Dielectric constant
Wall_Compensation_Dist = Thickness_Wall * (sqrt(Wall_Dieletric) - 1);       % Compensation distance

% --- Data Reading Parameters ---
FilePath = "D:\JoeyBG_Research_Production\TWR_Identity_Threat\RW_Set\RW_Datas\P1_Gun.NOV"; 
Total_Readin_Packages = 100;                                                % Frames to read
Fast_Time_Points = 3940;                                                    % ADC samples per chirp
Readin_DS_Ratio = 1;                                                        % Downsampling ratio reading
Process_DS_Ratio = 5;                                                       % Downsampling ratio processing
Total_DS_Ratio = Readin_DS_Ratio * Process_DS_Ratio;

% --- Radar Parameters ---
fc = data_info.f0;            
B = data_info.B;              
T = data_info.PRT;            
fs = data_info.fs;            
PRF_Effective = (1/T) / Total_DS_Ratio;                                     % Effective PRF

% --- Channel Selection ---
Target_Channel_Idx = 1; 

% --- Processing Parameters ---
FFT_Points = 4096;                                                          % Range FFT length
MTI_Step = 1;                                                               % MTI pulse canceller step
Win_Length = 20;                                                            % Sliding Window Length (Pulses)
Win_Stride = 2;                                                             % Sliding Window Stride (Pulses)
N_Doppler_FFT = 64;                                                         % Doppler FFT size
Intensity_Threshold_Ratio = 0.15;                                           % Visualization threshold
Min_Range_ROI = 0.5;                                                        % Display Min Range
Max_Range_ROI = 6.0;                                                        % Display Max Range

% --- Visualization Parameters ---
Font_Name = 'Palatino Linotype'; 
Font_Size_Basis = 12;
Font_Size_Title = 14;

%% Data Reading and Preprocessing
fprintf('Reading data file: %s ...\n', FilePath);

if ~isfile(FilePath)
    error('File not found: %s. Please check the path.', FilePath);
end

% Call the Novasky Reader Function
[adc_data_raw, ~] = Read_NOV(FilePath, Total_Readin_Packages, 8, 8, Fast_Time_Points, Readin_DS_Ratio);
adc_data = permute(adc_data_raw, [1 3 2]); % [FastTime, Channels, SlowTime]

[N_Fast, N_Channels_Total, N_Slow] = size(adc_data);
fprintf('Data Loaded: %d Samples, %d Channels, %d Frames\n', N_Fast, N_Channels_Total, N_Slow);

%% Signal Processing & RTD Generation
fprintf('Processing Channel %d into Complex RTD Feature...\n', Target_Channel_Idx);

% 1. Extract Raw Data
Raw_Ch = squeeze(adc_data(:, Target_Channel_Idx, :)); % [FastTime x SlowTime]

% 2. Range Compression
Range_Profile = fft(Raw_Ch, FFT_Points, 1);
Range_Profile = Range_Profile(1:FFT_Points/2, :); % Keep positive half

% 3. MTI Clutter Removal
Range_Profile_MTI = Range_Profile(:, MTI_Step+1:end) - Range_Profile(:, 1:end-MTI_Step);
Range_Profile_MTI = [Range_Profile_MTI, Range_Profile_MTI(:,end)]; % Pad to maintain size
[Num_Range_Bins, Num_Pulses_MTI] = size(Range_Profile_MTI);

% 4. Keystone Transform
fprintf('Applying Keystone Transform...\n');
RT_Keystone = zeros(size(Range_Profile_MTI));
slow_time_indices = 0:Num_Pulses_MTI-1;

% Calculate Frequency Axis for Baseband Data
Range_Resolution = c / (2 * B);
freq_axis_beat = (0:Num_Range_Bins-1)' * (fs / FFT_Points);
freq_axis_rf = fc + freq_axis_beat; 

for r = 1:Num_Range_Bins
    % Scaling factor
    f_curr = freq_axis_rf(r);
    scale_factor = f_curr / fc;
    
    % Interpolation
    t_new = slow_time_indices * scale_factor;
    % Use 'pchip' or 'spline' for smoother results on real data
    RT_Keystone(r, :) = interp1(slow_time_indices, Range_Profile_MTI(r, :), t_new, 'pchip', 0);
end

% 5. 3D Complex RTD Construction
fprintf('Constructing 3D Complex Cube...\n');
num_windows = floor((Num_Pulses_MTI - Win_Length) / Win_Stride) + 1;
RTD_Complex_Cube = zeros(Num_Range_Bins, N_Doppler_FFT, num_windows);

Doppler_Window = hamming(Win_Length);

for w = 1:num_windows
    idx_start = (w-1)*Win_Stride + 1;
    idx_end = idx_start + Win_Length - 1;
    
    segment = RT_Keystone(:, idx_start:idx_end);
    
    % Windowing
    segment_win = segment .* Doppler_Window';
    
    % FFT along slow time -> Doppler
    RD_Map = fftshift(fft(segment_win, N_Doppler_FFT, 2), 2);
    
    RTD_Complex_Cube(:, :, w) = RD_Map;
end

%% Visualization
fprintf('Visualizing Measured 3D Feature...\n');

% Axes Calculation
Range_Axis_Raw = ((0:Num_Range_Bins-1) * Range_Resolution);
Range_Axis_Physical = Range_Axis_Raw - Wall_Compensation_Dist; % Apply Wall Comp to Axis
Doppler_Axis = linspace(-PRF_Effective/2, PRF_Effective/2, N_Doppler_FFT);
Time_Axis = (0:num_windows-1) * Win_Stride * (1/PRF_Effective);

% ROI Cropping (Range)
[~, min_r] = min(abs(Range_Axis_Physical - Min_Range_ROI));
[~, max_r] = min(abs(Range_Axis_Physical - Max_Range_ROI));

RTD_ROI = RTD_Complex_Cube(min_r:max_r, :, :);
Range_Axis_ROI = Range_Axis_Physical(min_r:max_r);

% Magnitude & Normalization
RTD_Mag = abs(RTD_ROI);
RTD_Mag = RTD_Mag / max(RTD_Mag(:));

% Thresholding for 3D Scatter
threshold_val = quantile(RTD_Mag(:), 1 - Intensity_Threshold_Ratio);
[r_idx, d_idx, t_idx] = ind2sub(size(RTD_Mag), find(RTD_Mag > threshold_val));
intensities = RTD_Mag(RTD_Mag > threshold_val);

% Map Indices to Physical Units
P_range = Range_Axis_ROI(r_idx)';
P_doppler = Doppler_Axis(d_idx)';
P_time = Time_Axis(t_idx)';

figure('Name', 'Measured Complex RTD Feature', 'Color', 'w', 'Position', [100, 100, 1000, 600]);

% 3D Scatter Plot
scatter3(P_time, P_doppler, P_range, 20, intensities, 'filled', ...
    'MarkerFaceAlpha', 0.6);

% Aesthetics
colormap(MyColormap);
cb = colorbar;
cb.Label.String = 'Normalized Intensity';
cb.Label.FontName = Font_Name;
cb.Label.FontSize = Font_Size_Basis;

xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
ylabel('Doppler (Hz)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
zlabel('Range (m)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);

title(['Measured 3D RTD Feature - Channel ' num2str(Target_Channel_Idx)], 'FontName', Font_Name, 'FontSize', Font_Size_Title);

set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
grid on;
box on;

% View Adjustments
ylim([-PRF_Effective/2 PRF_Effective/2]);
zlim([Min_Range_ROI Max_Range_ROI]);
view(45, 30);

fprintf('Measured CV-RTD Generation Complete.\n');