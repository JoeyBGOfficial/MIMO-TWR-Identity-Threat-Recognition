%% Script for Measured Enhanced DTM Generation
% Original Author: Longzhen Tang, Shisheng Guo, Jiachen Li, Junda Zhu, Guolong Cui, Lingjiang Kong, and Xiaobo Yang.
% Reproduced By: JoeyBG.
% Date: 2025-12-29.
% Affiliate: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description:
%   1. Reads raw measured radar data .NOV using strict Read_NOV protocol.
%   2. Preprocesses: Range Compression, ROI Cutting, and Entropy-based Channel Selection.
%   3. Implements Two-Pulse Cancellation TPC and Average Cancellation AC on measured data.
%   4. Generates Doppler Time Maps DTM via STFT summation over range bins.
%   5. Performs Adaptive Weighted Fusion to generate Enhanced Spectrogram.
%   6. Visualizes TPC, AC, and Enhanced DTMs.

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
c = 3e8;                                                                    % Speed of light m/s
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

% --- Spectrogram & Fusion Parameters ---
STFT_Win_Size = 20;                                                         % Window size for STFT
STFT_Overlap = 18;                                                          % Overlap size
N_fft_doppler = 256;                                                        % Doppler FFT size
Fusion_Bandwidth_Sigma = 10;                                                % Sigma for Gaussian fusion mask

% --- Visualization Parameters ---
Font_Name = 'Palatino Linotype'; 
Font_Size_Basis = 12;
Font_Size_Title = 14;
if ~exist('CList', 'var')
    JoeyBG_Colormap = jet(256);
end
JoeyBG_Colormap_Flip = flip(CList); % Flip if necessary for standard DTM look

%% Section 1 Data Reading and Preprocessing
fprintf('Reading data file: %s ...\n', FilePath);

% 1.1 Read Data
% Returns [FastTime, SlowTime, Channels]
[adc_data_raw, ~] = Read_NOV(FilePath, Total_Readin_Packages, 8, 8, Fast_Time_Points, Readin_DS_Ratio);
% Permute to [FastTime, Channels, SlowTime] for consistency
adc_data = permute(adc_data_raw, [1 3 2]); 
[~, Total_Channels, Total_Frames_Read] = size(adc_data);

% 1.2 RTM Generation and Temporary MTI for Channel Selection
fprintf('Generating Range Profiles and Selecting Channel...\n');
% Range Compression
Range_Profile_Complex = fft(adc_data, FFT_Points, 1);
% Cut ROI
RTM_ROI_Raw = Range_Profile_Complex(Min_Bin_Idx:Max_Bin_Idx, :, :);

% MTI Filter
RTM_MTI_Temp = RTM_ROI_Raw(:, :, MTI_Step+1:end) - RTM_ROI_Raw(:, :, 1:end-MTI_Step);
[nRange, ~, Final_Frames] = size(RTM_MTI_Temp);

% 1.3 Channel Selection via Entropy
RTM_Mag_Stack = zeros(nRange, Final_Frames, Total_Channels);
entropies = zeros(1, Total_Channels);

for ch = 1:Total_Channels
    temp = abs(squeeze(RTM_MTI_Temp(:, ch, :)));
    RTM_Mag_Stack(:,:,ch) = mat2gray(temp);
    entropies(ch) = entropy(im2uint8(RTM_Mag_Stack(:,:,ch)));
end

[min_entropy_val, Ref_Idx] = min(entropies);
fprintf('  > Selected Channel: %d (Entropy: %.4f)\n', Ref_Idx, min_entropy_val);

% Extract the best channel RAW complex data (Before MTI)
% Dimensions: [Range, SlowTime]
Best_RTM_Complex = squeeze(RTM_ROI_Raw(:, Ref_Idx, :)); 

%% Section 2 Clutter Suppression (TPC & AC)
fprintf('Applying TPC and AC Clutter Suppression...\n');

% 2.1 Two-Pulse Cancellation TPC
% Subtract consecutive pulses
RTM_TPC = Best_RTM_Complex(:, 2:end) - Best_RTM_Complex(:, 1:end-1);

% 2.2 Average Cancellation AC
% Subtract mean across slow time
Mean_Clutter = mean(Best_RTM_Complex, 2);
RTM_AC = Best_RTM_Complex - Mean_Clutter;
% Truncate last frame to match TPC dimensions
RTM_AC = RTM_AC(:, 1:end-1);

%% Section 3 Enhanced Spectrogram Generation
fprintf('Generating Enhanced Doppler Spectrograms...\n');

STFT_Window = hamming(STFT_Win_Size);

% Function to generate DTM:
% 1. Transpose Input to [SlowTime, Range] for stft function (requires time in rows usually)
% 2. stft returns [Freq, Time, Range]
% 3. abs() -> magnitude
% 4. sum() along dim 3 (Range) to collapse range bins
% 5. squeeze() -> [Freq, Time]
generate_DTM = @(RTM_Input) squeeze(sum(abs(stft(RTM_Input.', PRF_Effective, ...
    'Window', STFT_Window, 'OverlapLength', STFT_Overlap, 'FFTLength', N_fft_doppler)), 3));

% 3.1 Generate Individual Spectrograms
DTM_TPC = generate_DTM(RTM_TPC);
DTM_AC  = generate_DTM(RTM_AC);

% Get Time and Freq vectors
[~, f_vec, t_vec] = stft(RTM_TPC(1,:), PRF_Effective, 'Window', STFT_Window, ...
    'OverlapLength', STFT_Overlap, 'FFTLength', N_fft_doppler);

% 3.2 Normalization 
DTM_TPC_Norm = DTM_TPC / max(DTM_TPC(:));
DTM_AC_Norm  = DTM_AC / max(DTM_AC(:));

% 3.3 Adaptive Weighted Fusion
freq_indices = linspace(-PRF_Effective/2, PRF_Effective/2, N_fft_doppler)';

% AC Weight: High at Low Freq
Weight_AC_Vec = exp(- (freq_indices.^2) / (2 * Fusion_Bandwidth_Sigma^2));
% TPC Weight: High at High Freq
Weight_TPC_Vec = 1 - Weight_AC_Vec;

% Expand to match Time dimension
Weight_AC_Map = repmat(Weight_AC_Vec, 1, length(t_vec));
Weight_TPC_Map = repmat(Weight_TPC_Vec, 1, length(t_vec));

% Fuse
DTM_Enhanced = (Weight_TPC_Map .* DTM_TPC_Norm) + (Weight_AC_Map .* DTM_AC_Norm);
DTM_Enhanced = DTM_Enhanced / max(DTM_Enhanced(:));

%% Section 4 Visualization
fprintf('Visualizing Results...\n');
figure('Name', 'Measured ADFE Analysis', 'Color', 'w', 'Position', [100, 100, 1400, 500]);

% Plot 1: Two-Pulse Cancellation TPC
subplot(1, 3, 1);
imagesc(t_vec, f_vec, DTM_TPC_Norm);
colormap(JoeyBG_Colormap_Flip); % Use loaded colormap
caxis([0 1]);
title('(a) Measured Two-Pulse Cancellation', 'FontName', Font_Name, 'FontSize', Font_Size_Title);
xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
ylabel('Doppler (Hz)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'YDir', 'normal');
colorbar;

% Plot 2: Average Cancellation AC
subplot(1, 3, 2);
imagesc(t_vec, f_vec, DTM_AC_Norm);
colormap(JoeyBG_Colormap_Flip);
caxis([0 1]);
title('(b) Measured Average Cancellation', 'FontName', Font_Name, 'FontSize', Font_Size_Title);
xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
ylabel('Doppler (Hz)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'YDir', 'normal');
colorbar;

% Plot 3: Enhanced Spectrogram
subplot(1, 3, 3);
imagesc(t_vec, f_vec, DTM_Enhanced);
colormap(JoeyBG_Colormap_Flip);
caxis([0 1]);
title('(c) Measured Enhanced Spectrogram', 'FontName', Font_Name, 'FontSize', Font_Size_Title);
xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
ylabel('Doppler (Hz)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'YDir', 'normal');
cb = colorbar;
cb.Label.String = 'Normalized Amplitude';
cb.Label.FontName = Font_Name;
cb.Label.FontSize = Font_Size_Basis;

fprintf('Processing Complete.\n');