%% Script for Measured DTM Generation with Multistatic Bio Radar
% Original Author: Yimeng Zhao, Yong Jia, Dong Huang, Li Zhang, Yao Zheng, Jianqi Wang, and Fugui Qi.
% Improved By: JoeyBG.
% Date: 2025-12-28.
% Affiliate: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description:
%   1 Reads raw measured radar data .NOV using Read_NOV function.
%   2 Preprocesses data including Range Compression, ROI Selection, and MTI.
%   3 Generates DTM spectrograms for all 8 channels via STFT.
%   4 Visualizes the 8-channel DTMs using normalized amplitude.

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
Total_Readin_Packages = 100;                                                % Increased packages for better DTM duration
Fast_Time_Points = 3940;      
Readin_DS_Ratio = 1;                                                        % Reading DS ratio
Process_DS_Ratio = 5;                                                       % Processing DS ratio
Total_DS_Ratio = Readin_DS_Ratio * Process_DS_Ratio;

% --- Radar System Parameters ---
c = 3e8;                                                                    % Speed of light (m/s)
fc = data_info.f0;
B = data_info.B;              
T = data_info.PRT;            
fs = data_info.fs;            
PRF_Raw = 1/T;                
PRF_Effective = PRF_Raw / Total_DS_Ratio;                                   % Effective PRF (Hz)

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

% --- Spectrogram Parameters ---
STFT_Win_Size = 20;                                                         % Window size for DTM
STFT_Overlap = 18;                                                          % Overlap size
N_fft_doppler = 512;                                                        % Doppler FFT size
STFT_Window = hamming(STFT_Win_Size);

% --- Visualization Parameters ---
Font_Name = 'Palatino Linotype'; 
Font_Size_Basis = 10;
Font_Size_Title = 12;
if ~exist('CList', 'var')
    JoeyBG_Colormap = jet(256);
end
JoeyBG_Colormap_Flip = flip(CList);

%% Section 1 Data Reading and Preprocessing
fprintf('Reading data file: %s ...\n', FilePath);

% 1.1 Read Data
% Returns [FastTime, SlowTime, Channels]
[adc_data_raw, ~] = Read_NOV(FilePath, Total_Readin_Packages, 8, 8, Fast_Time_Points, Readin_DS_Ratio);
% Permute to [FastTime, Channels, SlowTime] for processing loop
adc_data = permute(adc_data_raw, [1 3 2]); 
[~, Total_Channels, Total_Frames_Read] = size(adc_data);

%% Section 2 DTM Generation and Visualization
fprintf('Processing 8-Channel DTMs...\n');

figure('Name', 'Measured 8-Channel DTMs', 'Color', 'w', 'Position', [50, 50, 1400, 700]);
tiledlayout(2, 4, 'Padding', 'compact', 'TileSpacing', 'compact');

for k = 1:min(8, Total_Channels)
    % 2.1 Range Profile Extraction
    Raw_Single_Channel = squeeze(adc_data(:, k, :)); % [FastTime, SlowTime]
    
    % Range Compression
    Range_Profile_Complex = fft(Raw_Single_Channel, FFT_Points, 1);
    
    % ROI Selection
    RTM_ROI_Raw = Range_Profile_Complex(Min_Bin_Idx:Max_Bin_Idx, :);
    
    % 2.2 Wall Clutter Removal via MTI
    % Subtract previous pulse from current pulse
    RTM_MTI = RTM_ROI_Raw(:, 2:end) - RTM_ROI_Raw(:, 1:end-1);
    
    % 2.3 Sum along range bins to get time series for DTM
    dtm_time_series = sum(RTM_MTI, 1);
    
    % 2.4 STFT
    [S_stft, f_vec, t_vec] = stft(dtm_time_series, PRF_Effective, ...
        'Window', STFT_Window, 'OverlapLength', STFT_Overlap, 'FFTLength', N_fft_doppler);
    
    % Normalize Amplitude
    Mag_DTM = abs(S_stft);
    Norm_DTM = Mag_DTM / max(Mag_DTM(:));
    
    % 2.5 Visualization
    nexttile;
    imagesc(t_vec, f_vec, Norm_DTM);
    colormap(JoeyBG_Colormap_Flip);
    axis xy;
    
    % Styling
    title(['Channel ' num2str(k)], 'FontName', Font_Name, 'FontSize', Font_Size_Title);
    
    if k > 4
        xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
    end
    if mod(k, 4) == 1
        ylabel('Doppler (Hz)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
    end
    
    set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
    ylim([-PRF_Effective/2 PRF_Effective/2]); 
end

fprintf('Processing Complete.\n');