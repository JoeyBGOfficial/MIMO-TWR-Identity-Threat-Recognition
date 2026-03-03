%% Data Reading and Processing of RW_Datas based on PSNR Screening 
% Former Author: Wang Tao, JoeyBG.
% Improved By: JoeyBG.
% Date: 2025-12-05.
% Affiliate: National University of Defense Technology, Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Introduction:
% This script achieves the complete workflow of reading raw radar data, preprocessing, and visualization for the Novasky MIMO radar system.
% The program reads the raw binary ADC data and performs the following steps:
%   1. Parameter Initialization: Loads radar configurations and defines physical constants.
%   2. Wall Compensation: Calculates the electromagnetic delay caused by the wall to calibrate the zero-range point to the back of the wall.
%   3. Data Reading: Unpacks the binary file with specified downsampling ratios.
%   4. RTM Generation: Performs pulse compression via FFT and Moving Target Indication to generate the Range-Time Map for all 64 channels.
%   5. DTM Generation: Applies STFT to the range-aggregated signal to generate the Doppler-Time Map.
%   6. RDM Generation: Applies 2D-FFT to generate Range-Doppler Map.
%   7. Enhancement: For RTM, DTM, and RDM, select reference channel with minimal entropy, select channels with PSNR > Threshold, and fuse using wfusimg.
%   8. Visualization: Plots Reference vs Enhanced for RTM, DTM, and RDM in a 3x2 grid.

%% Initialization of MATLAB Script
clear all;
close all;
clc;
disp('---------- © Author: JoeyBG © ----------');

% Load configurations
try
    load("JoeyBG_Colormap.mat");                                            % My favorite colormap
    load("Array_Configurations\data_info.mat");                             % Radar configuration
catch
    warning('Radar configuration file is missed! Use default instead.');
    if ~exist('data_info','var')
        data_info.f0 = 2.5e9; 
        data_info.B = 1.0e9; 
        data_info.PRT = 1e-3; 
        data_info.fs = 4e6; 
        data_info.antenna_posi = zeros(1,1,1);
    end
end

%% Parameter Definition
% Speed of light
c = 3e8;

% Wall parameters
Thickness_Wall = 0.12;    
Wall_Dieletric = 6;       
Wall_Compensation_Dist = Thickness_Wall * (sqrt(Wall_Dieletric) - 1);       % Extra electromagnetic path caused by the wall

% Data reading parameters
FilePath = "RW_Datas\P1_Gun.NOV";
Total_Readin_Packages = 100;                                                % Maximum packages for reading
Fast_Time_Points = 3940;      
Readin_DS_Ratio = 1;                                                        % Downsampling ratio for data reading
Process_DS_Ratio = 5;                                                       % Downsampling ratio for data preprocessing
Total_DS_Ratio = Readin_DS_Ratio * Process_DS_Ratio;

% Radar parameters
B = data_info.B;              
T = data_info.PRT;            
fs = data_info.fs;            
PRF_Raw = 1/T;                
PRF_Effective = PRF_Raw / Total_DS_Ratio;                                   % Effective PRF value

% Calculate the range axis
FFT_Points = Fast_Time_Points;          
Range_Resolution = c / (2 * B);
Range_Axis_Raw = ((0:FFT_Points/2-1) * Range_Resolution) / 2;               % Range axis calculation based on bandwidth and FFT points

% Wall compensation
Range_Axis_Physical = Range_Axis_Raw - Wall_Compensation_Dist;

% Select ROI for RTM/RDM generation and visualization
Min_Range_ROI = 0;                                                          
Max_Range_ROI = 6;                                                                                                                                  
[~, Min_Bin_Idx] = min(abs(Range_Axis_Physical - Min_Range_ROI));
[~, Max_Bin_Idx] = min(abs(Range_Axis_Physical - Max_Range_ROI));
Range_Axis_Show = Range_Axis_Physical(Min_Bin_Idx:Max_Bin_Idx);             % Cut the range axis
MTI_Step = 1;                                                               % Step of MTI for clutter cancelling

% STFT / Doppler parameters
Window_Len_Sec = 0.1;                                                       
Window_Len_Points = floor(Window_Len_Sec * PRF_Effective);                  
Overlap_Ratio = 0.9;                                                        
Overlap_Points = floor(Window_Len_Points * Overlap_Ratio);                  
STFT_FFT_Len = 1024;                                                        % FFT points for Doppler dimension
STFT_Win = hamming(Window_Len_Points, "periodic");                          

% Fusion parameters
wname = 'db4';                                                              
wmethod = 'mean';                                                           
level = 2;                                                                  
PSNR_Threshold_RTM = 25;                                                    % PSNR threshold for RTM
PSNR_Threshold_DTM = 20;                                                    % PSNR threshold for DTM
PSNR_Threshold_RDM = 30;                                                    % PSNR threshold for RDM

% Visualization parameters
Font_Name = 'Palatino Linotype';                                            
Font_Size_Basis = 12;                                                       
Font_Size_Axis = 14;                                                        
Font_Size_Title = 16;                                                       
Font_Weight_Basis = 'normal';                                               
Font_Weight_Axis = 'normal';                                                
Font_Weight_Title = 'bold';                                                 

%% Data Reading
fprintf('Read the data file: %s ...\n', FilePath);

% Use Read_NOV function for data reading (8 Tx * 8 Rx = 64 Channels)
[adc_data_raw, ~] = Read_NOV(FilePath, Total_Readin_Packages, 8, 8, Fast_Time_Points, Readin_DS_Ratio);
adc_data = permute(adc_data_raw, [1 3 2]); % Permute to [FastTime, Channels, SlowTime]
[~, Total_Channels, Total_Frames_Read] = size(adc_data);
fprintf('Total Frames: %d, Total Channels: %d\n', Total_Frames_Read, Total_Channels);

%% RTM Generation for All Channels
fprintf('Generate RTMs for all channels...\n');

% Dechirp-based pulse compression
Range_Profile_Complex = fft(adc_data, FFT_Points, 1);
RTM_ROI_Raw = Range_Profile_Complex(Min_Bin_Idx:Max_Bin_Idx, :, :); % Cut the ROI

% MTI filtering [Range, Channels, Time]
RTM_MTI = RTM_ROI_Raw(:, :, MTI_Step+1:end) - RTM_ROI_Raw(:, :, 1:end-MTI_Step);
[nRange, ~, Final_Frames] = size(RTM_MTI);
Time_Axis_RTM = (0 : Final_Frames - MTI_Step - 1) * (T * Total_DS_Ratio); 

%% RTM Feature Fusion Process
fprintf('Start RTM Fusion...\n');

% Pre-process RTM Magnitudes
RTM_Mag_Stack = zeros(nRange, Final_Frames, Total_Channels);
for ch = 1:Total_Channels
    temp = abs(squeeze(RTM_MTI(:, ch, :)));
    RTM_Mag_Stack(:,:,ch) = mat2gray(temp); % Normalize 
end

% Calculate entropy and find reference
entropies = zeros(1, Total_Channels);
for ch = 1:Total_Channels
    entropies(ch) = entropy(im2uint8(RTM_Mag_Stack(:,:,ch)));
end
[min_entropy_val, Ref_Idx_RTM] = min(entropies);
fprintf('  > RTM Reference Channel: %d (Entropy: %.4f)\n', Ref_Idx_RTM, min_entropy_val);
Reference_RTM = RTM_Mag_Stack(:,:,Ref_Idx_RTM);

% Calculate PSNR and select channels
selected_indices_RTM = [];
for ch = 1:Total_Channels
    if ch == Ref_Idx_RTM, continue; end
    p_val = psnr(RTM_Mag_Stack(:,:,ch), Reference_RTM);
    if p_val > PSNR_Threshold_RTM
        selected_indices_RTM = [selected_indices_RTM, ch];
    end
end
fprintf('  > RTM Fusion Candidates: %d channels\n', length(selected_indices_RTM));

% Wavelet feature fusion
Enhanced_RTM = Reference_RTM;
for idx = selected_indices_RTM
    Enhanced_RTM = wfusimg(Enhanced_RTM, RTM_Mag_Stack(:,:,idx), wname, level, wmethod, wmethod);
end

%% DTM Generation for All Channels
fprintf('Generate DTMs for all channels...\n');

if Window_Len_Points > length(Time_Axis_RTM)
    Window_Len_Points = length(Time_Axis_RTM) - 2;
    STFT_Win = hamming(Window_Len_Points, "periodic");
    Overlap_Points = floor(Window_Len_Points * Overlap_Ratio);
end

% Placeholder
temp_sig = sum(squeeze(RTM_MTI(:, 1, :)), 1);
[s_temp, f_freq, t_time] = stft(temp_sig, PRF_Effective, ...
    'Window', STFT_Win, 'OverlapLength', Overlap_Points, ...
    'FFTLength', STFT_FFT_Len); 
DTM_Time_Axis = t_time;
DTM_Mag_Stack = zeros(length(f_freq), length(t_time), Total_Channels);

for ch = 1:Total_Channels
    Raw_Signal_1D = sum(squeeze(RTM_MTI(:, ch, :)), 1); 
    [S_stft, ~, ~] = stft(Raw_Signal_1D, PRF_Effective, ...
        'Window', STFT_Win, 'OverlapLength', Overlap_Points, ...
        'FFTLength', STFT_FFT_Len);
    DTM_Mag_Stack(:,:,ch) = mat2gray(abs(S_stft)); 
end

%% DTM Fusion Process
fprintf('Start DTM Fusion...\n');

entropies_dtm = zeros(1, Total_Channels);
for ch = 1:Total_Channels
    entropies_dtm(ch) = entropy(im2uint8(DTM_Mag_Stack(:,:,ch)));
end
[min_entropy_dtm, Ref_Idx_DTM] = min(entropies_dtm);
fprintf('  > DTM Reference Channel: %d (Entropy: %.4f)\n', Ref_Idx_DTM, min_entropy_dtm);
Reference_DTM = DTM_Mag_Stack(:,:,Ref_Idx_DTM);

selected_indices_DTM = [];
for ch = 1:Total_Channels
    if ch == Ref_Idx_DTM, continue; end
    p_val = psnr(DTM_Mag_Stack(:,:,ch), Reference_DTM);
    if p_val > PSNR_Threshold_DTM
        selected_indices_DTM = [selected_indices_DTM, ch];
    end
end
fprintf('  > DTM Fusion Candidates: %d channels\n', length(selected_indices_DTM));

Enhanced_DTM = Reference_DTM;
for idx = selected_indices_DTM
    Enhanced_DTM = wfusimg(Enhanced_DTM, DTM_Mag_Stack(:,:,idx), wname, level, wmethod, wmethod);
end

%% RDM Generation for All Channels
fprintf('Generate RDMs for all channels...\n');

% Calculate Doppler Axis
Doppler_Axis = linspace(-PRF_Effective/2, PRF_Effective/2, STFT_FFT_Len);

% Initialize Stack [Range, Doppler, Channels]
RDM_Mag_Stack = zeros(nRange, STFT_FFT_Len, Total_Channels);

% Window function for Slow Time dimension to reduce sidelobes
Win_Doppler = hamming(Final_Frames); 

for ch = 1:Total_Channels
    % Extract Range-SlowTime matrix for current channel [Range x Time]
    Current_RTM_Cpx = squeeze(RTM_MTI(:, ch, :));
    
    % Apply windowing along Slow Time (columns)
    Current_RTM_Cpx_Win = Current_RTM_Cpx .* Win_Doppler.';
    
    % Perform FFT along Slow Time dimension
    RDM_Complex = fftshift(fft(Current_RTM_Cpx_Win, STFT_FFT_Len, 2), 2);
    
    % Compute Magnitude and Normalize
    RDM_Mag_Stack(:,:,ch) = mat2gray(abs(RDM_Complex));
end

%% RDM Fusion Process
fprintf('Start RDM Fusion...\n');

% Calculate entropy and find reference
entropies_rdm = zeros(1, Total_Channels);
for ch = 1:Total_Channels
    % Use im2uint8 to create standard histogram
    entropies_rdm(ch) = entropy(im2uint8(RDM_Mag_Stack(:,:,ch)));
end
[min_entropy_rdm, Ref_Idx_RDM] = min(entropies_rdm);
fprintf('  > RDM Reference Channel: %d (Entropy: %.4f)\n', Ref_Idx_RDM, min_entropy_rdm);

% Record reference channel for RDM
Reference_RDM = RDM_Mag_Stack(:,:,Ref_Idx_RDM);

% Calculate PSNR and filtering
selected_indices_RDM = [];
for ch = 1:Total_Channels
    if ch == Ref_Idx_RDM, continue; end
    p_val = psnr(RDM_Mag_Stack(:,:,ch), Reference_RDM);
    if p_val > PSNR_Threshold_RDM
        selected_indices_RDM = [selected_indices_RDM, ch];
    end
end
fprintf('  > RDM Fusion Candidates: %d channels\n', length(selected_indices_RDM));

% Wavelet feature fusion
Enhanced_RDM = Reference_RDM;
for idx = selected_indices_RDM
    Enhanced_RDM = wfusimg(Enhanced_RDM, RDM_Mag_Stack(:,:,idx), wname, level, wmethod, wmethod);
end

%% Visualization
fprintf('Start visualizing...\n');
figure('Name', 'Multi-Domain Fusion Analysis', 'Color', 'w', 'Position', [100, 50, 1200, 900]);

get_log_display = @(img) log((img - min(img(:)))...
    / (max(img(:)) - min(img(:))) + 1e-6);

% ================= ROW 1: RTM =================
% 1. Reference RTM
subplot(3, 2, 1);
imagesc(Time_Axis_RTM, Range_Axis_Show, get_log_display(Reference_RTM));
axis xy; 
if exist('CList_Flip', 'var'), colormap(gca, CList_Flip); else, colormap(gca, jet); end
colorbar; clim([-3, 0]); 
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'LineWidth', 1.5);
title(['Ref RTM (Ch ', num2str(Ref_Idx_RTM), ')'], 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
xlabel('Time (s)', 'FontSize', Font_Size_Axis); ylabel('Range (m)', 'FontSize', Font_Size_Axis);
ylim([Min_Range_ROI, Max_Range_ROI]);

% 2. Enhanced RTM
subplot(3, 2, 2);
imagesc(Time_Axis_RTM, Range_Axis_Show, get_log_display(Enhanced_RTM));
axis xy; 
if exist('CList_Flip', 'var'), colormap(gca, CList_Flip); else, colormap(gca, jet); end
colorbar; clim([-3, 0]); 
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'LineWidth', 1.5);
title('Enhanced RTM', 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
xlabel('Time (s)', 'FontSize', Font_Size_Axis); ylabel('Range (m)', 'FontSize', Font_Size_Axis);
ylim([Min_Range_ROI, Max_Range_ROI]);

% ================= ROW 2: DTM =================
% 3. Reference DTM
subplot(3, 2, 3);
imagesc(DTM_Time_Axis, f_freq, get_log_display(Reference_DTM));
axis xy;
if exist('CList_Flip', 'var'), colormap(gca, CList_Flip); else, colormap(gca, jet); end
colorbar; clim([-3, 0]); 
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'LineWidth', 1.5);
title(['Ref DTM (Ch ', num2str(Ref_Idx_DTM), ')'], 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
xlabel('Time (s)', 'FontSize', Font_Size_Axis); ylabel('Doppler (Hz)', 'FontSize', Font_Size_Axis);
ylim([-100, 100]);

% 4. Enhanced DTM
subplot(3, 2, 4);
imagesc(DTM_Time_Axis, f_freq, get_log_display(Enhanced_DTM));
axis xy;
if exist('CList_Flip', 'var'), colormap(gca, CList_Flip); else, colormap(gca, jet); end
colorbar; clim([-3, 0]); 
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'LineWidth', 1.5);
title('Enhanced DTM', 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
xlabel('Time (s)', 'FontSize', Font_Size_Axis); ylabel('Doppler (Hz)', 'FontSize', Font_Size_Axis);
ylim([-100, 100]);

% ================= ROW 3: RDM =================
% 5. Reference RDM
subplot(3, 2, 5);
imagesc(Doppler_Axis, Range_Axis_Show, get_log_display(Reference_RDM));
axis xy;
if exist('CList_Flip', 'var'), colormap(gca, CList_Flip); else, colormap(gca, jet); end
colorbar; clim([-3, 0]); 
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'LineWidth', 1.5);
title(['Ref RDM (Ch ', num2str(Ref_Idx_RDM), ')'], 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
xlabel('Doppler (Hz)', 'FontSize', Font_Size_Axis); ylabel('Range (m)', 'FontSize', Font_Size_Axis);
xlim([-100, 100]); ylim([Min_Range_ROI, Max_Range_ROI]);

% 6. Enhanced RDM
subplot(3, 2, 6);
imagesc(Doppler_Axis, Range_Axis_Show, get_log_display(Enhanced_RDM));
axis xy;
if exist('CList_Flip', 'var'), colormap(gca, CList_Flip); else, colormap(gca, jet); end
colorbar; clim([-3, 0]); 
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'LineWidth', 1.5);
title('Enhanced RDM', 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
xlabel('Doppler (Hz)', 'FontSize', Font_Size_Axis); ylabel('Range (m)', 'FontSize', Font_Size_Axis);
xlim([-100, 100]); ylim([Min_Range_ROI, Max_Range_ROI]);

fprintf('Visualization completes!\n');