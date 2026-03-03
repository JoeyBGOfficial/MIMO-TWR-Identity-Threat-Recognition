%% Script for Measured RTM Generation with Novasky MIMO Radar
% Original Author: Renming Liu, Yan Tang, Shaoming Zhang, Yusheng Li, and Jianmei Wang.
% Reproduced By: JoeyBG.
% Date: 2025-12-25.
% Affiliation: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description:
%   1 Reads raw binary radar data .NOV using Read_NOV function.
%   2 Extracts data for 4 specific channels to mimic 2x2 MIMO topology.
%   3 Performs Wall Compensation, Pulse Compression FFT, and MTI Clutter Removal.
%   4 Visualizes the Range-Time Maps RTM for the selected 4 channels in a 2x2 grid.

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
    warning('Configuration files missed Using defaults and Jet colormap');
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
FilePath = "D:\JoeyBG_Research_Production\TWR_Identity_Threat\RW_Set\RW_Datas\P1_Gun.NOV"; % Path to measured data
Total_Readin_Packages = 100;                                                % Number of frames to read
Fast_Time_Points = 3940;                                                    % ADC samples per chirp
Readin_DS_Ratio = 1;                                                        % Downsampling ratio reading
Process_DS_Ratio = 5;                                                       % Downsampling ratio processing
Total_DS_Ratio = Readin_DS_Ratio * Process_DS_Ratio;

% --- Radar Parameters ---
B = data_info.B;              
T = data_info.PRT;            
fs = data_info.fs;            
PRF_Effective = (1/T) / Total_DS_Ratio;                                     % Effective PRF

% --- Channel Selection ---
Selected_Channels = [1, 18, 36, 54]; 

% --- Processing Parameters ---
FFT_Points = 4096;                                                          % FFT length
MTI_Step = 1;                                                               % MTI pulse canceller step
Min_Range_ROI = 0;                                                          % Display Range Min (m)
Max_Range_ROI = 6;                                                          % Display Range Max (m)

% --- Visualization Parameters ---
Font_Name = 'Palatino Linotype'; 
Font_Size_Basis = 12;
Font_Size_Title = 14;

%% Data Reading and Preprocessing
fprintf('Reading data file: %s ...\n', FilePath);

% Check if file exists
if ~isfile(FilePath)
    error('File not found: %s. Please check the path.', FilePath);
end

% Call the Novasky Reader Function
[adc_data_raw, ~] = Read_NOV(FilePath, Total_Readin_Packages, 8, 8, Fast_Time_Points, Readin_DS_Ratio);
adc_data = permute(adc_data_raw, [1 3 2]); 

[N_Fast, N_Channels_Total, N_Slow] = size(adc_data);
fprintf('Data Loaded: %d Samples, %d Channels, %d Frames\n', N_Fast, N_Channels_Total, N_Slow);

%% Signal Processing Calculation
fprintf('Processing RTMs for selected channels...\n');

% 1. Range Axis Calculation
Range_Resolution = c / (2 * B);
Range_Axis_Raw = ((0:FFT_Points/2-1) * Range_Resolution) / 2;
Range_Axis_Physical = Range_Axis_Raw - Wall_Compensation_Dist;

% 2. ROI Indexing
[~, Min_Bin_Idx] = min(abs(Range_Axis_Physical - Min_Range_ROI));
[~, Max_Bin_Idx] = min(abs(Range_Axis_Physical - Max_Range_ROI));
Range_Axis_Show = Range_Axis_Physical(Min_Bin_Idx:Max_Bin_Idx);
Time_Axis = (0 : N_Slow - MTI_Step - 1) / PRF_Effective;

% 3. Process Selected Channels
RTM_Cell = cell(1, 4);

for k = 1:4
    ch_idx = Selected_Channels(k);
    
    % Extract Raw Data for Channel [FastTime, SlowTime]
    Raw_Ch = squeeze(adc_data(:, ch_idx, :));
    
    % Pulse Compression FFT
    Range_Profile = fft(Raw_Ch, FFT_Points, 1);
    
    % Cut ROI
    RTM_ROI = Range_Profile(Min_Bin_Idx:Max_Bin_Idx, :);
    
    % MTI Clutter Removal
    RTM_MTI = RTM_ROI(:, MTI_Step+1:end) - RTM_ROI(:, 1:end-MTI_Step);
    
    % Magnitude and Normalization
    RTM_Mag = abs(RTM_MTI);
    RTM_Norm = mat2gray(RTM_Mag);
    
    % Log Enhancement for Visualization
    RTM_Log = log((RTM_Norm - min(RTM_Norm(:))) / (max(RTM_Norm(:)) - min(RTM_Norm(:))) + eps);
    
    RTM_Cell{k} = RTM_Log;
end

%% Visualization
fprintf('Visualizing Results...\n');
figure('Name', 'Measured MIMO RTM Visualization', 'Color', 'w', 'Position', [100, 50, 1000, 800]);

for k = 1:4
    subplot(2, 2, k);
    
    % Plot RTM
    imagesc(Time_Axis, Range_Axis_Show, RTM_Cell{k}); 
    axis xy;
    
    % Aesthetics
    set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis);
    colormap(gca, MyColormap); 
    colorbar; 
    clim([-5 0]); % Adjust contrast limits as needed for measured data
    
    % Labels
    title(['Measured RTM Channel ' num2str(Selected_Channels(k))], 'FontName', Font_Name, 'FontSize', Font_Size_Title);
    xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
    ylabel('Range (m)', 'FontName', Font_Name, 'FontSize', Font_Size_Basis); 
    ylim([Min_Range_ROI Max_Range_ROI]);
end

fprintf('Measured Data Processing Complete.\n');