%% Batch Processing Script: Measured DTM Dataset Generator
% Original Author: Yimeng Zhao, Yong Jia, Dong Huang, Li Zhang, Yao Zheng, Jianqi Wang, and Fugui Qi.
% Improved By: JoeyBG.
% Date: 2025-12-28.
% Affiliate: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description:
%   1. Processes raw radar data (.NOV) from 'RW_Datas' folder in batches.
%   2. Uses a sliding window to slice data.
%   3. For each slice:
%      a. Generates RTM and performs MTI.
%      b. Extracts all 8 channels.
%      c. Computes STFT to generate DTM spectrograms for each channel.
%      d. Normalizes and saves the DTMs into channel-specific folders.
%   4. Output Structure: Measured_DTMSet_Channel{k}/{ClassName}/{Filename}.mat.

%% Initialization
clear all; 
close all; 
clc; 
disp('---------- © Author: JoeyBG © ----------');

%% Parameter Definitions
% --- File and Directory Settings ---
Data_Folder = "D:\JoeyBG_Research_Production\TWR_Identity_Threat\RW_Set\RW_Datas";
Output_Root_Base = "Measured_DTMSet_Channel";

File_List = [
    "P1_Nogun.NOV", "P1_Gun.NOV", ...
    "P2_Nogun.NOV", "P2_Gun.NOV", ...
    "P3_Nogun.NOV", "P3_Gun.NOV", ...
    "P4_Nogun.NOV", "P4_Gun.NOV"
];

% --- Reading Parameters ---
Window_Packets = 100;                                                       % Duration for one DTM sample
Step_Packets = 12;                                                          % Sliding step
Fast_Time_Points = 3940;                                                    
Rx_Num = 8;
Tx_Num = 8;
Header_Len = 60;
Bytes_Per_Int16 = 2;
Total_Channels_To_Process = 8;                                              % We need 8 channels for MSMV-DAN

% Block Calculation for fseek
DS_Read = 1; 
Points_Per_Block = Rx_Num * (Fast_Time_Points + Header_Len) * Tx_Num * DS_Read;
Bytes_Per_Block = Points_Per_Block * Bytes_Per_Int16;
Bytes_Per_Packet_Arg = Bytes_Per_Block * 2;                                 % solving_num = packet*2

% --- Radar Physics Parameters ---
c = 3e8;
try
    load("Array_Configurations\data_info.mat");
catch
    data_info.B = 1.0e9; 
    data_info.PRT = 1e-3; 
    data_info.fs = 4e6; 
end

B = data_info.B;
T = data_info.PRT;
Process_DS_Ratio = 5;
Total_DS_Ratio = DS_Read * Process_DS_Ratio;
PRF_Effective = (1/T) / Total_DS_Ratio;                                     % Effective PRF

% --- Wall Compensation ---
Thickness_Wall = 0.12;    
Wall_Dieletric = 6;       
Wall_Compensation_Dist = Thickness_Wall * (sqrt(Wall_Dieletric) - 1);

% --- Range Axis and ROI ---
FFT_Points = Fast_Time_Points;
Range_Resolution = c / (2 * B);
Range_Axis_Raw = ((0:FFT_Points/2-1) * Range_Resolution) / 2;
Range_Axis_Physical = Range_Axis_Raw - Wall_Compensation_Dist;

Min_Range_ROI = 0.5;                                                        % ROI Start (m)
Max_Range_ROI = 6.0;                                                        % ROI End (m)
[~, Min_Bin_Idx] = min(abs(Range_Axis_Physical - Min_Range_ROI));
[~, Max_Bin_Idx] = min(abs(Range_Axis_Physical - Max_Range_ROI));

% --- DTM Spectrogram Parameters ---
MTI_Step = 1;
STFT_Win_Size = 20;                                                         % Window size
STFT_Overlap = 18;                                                          % Overlap
N_fft_doppler = 256;                                                        % Doppler FFT size
STFT_Window = hamming(STFT_Win_Size);

%% Create Directory Structure
% Create folders for Channel 1 to 8
for ch = 1:Total_Channels_To_Process
    ch_root = sprintf('%s%d', Output_Root_Base, ch);
    if ~exist(ch_root, 'dir'), mkdir(ch_root); end
end

%% Main Processing Loop
fprintf('Starting Measured DTM Dataset Generation...\n');

for f_idx = 1:length(File_List)
    current_file_name = File_List(f_idx);
    full_path = fullfile(Data_Folder, current_file_name);
    
    [~, name_no_ext, ~] = fileparts(current_file_name);
    
    % Ensure subdirectories exist for this class in all channel folders
    for ch = 1:Total_Channels_To_Process
        ch_root = sprintf('%s%d', Output_Root_Base, ch);
        cls_dir = fullfile(ch_root, name_no_ext);
        if ~exist(cls_dir, 'dir'), mkdir(cls_dir); end
    end
    
    fprintf('Processing File (%d/%d): %s\n', f_idx, length(File_List), current_file_name);
    
    fid = fopen(full_path, 'r');
    if fid == -1
        warning('Cannot open file: %s', full_path);
        continue;
    end
    
    % Check file size
    fseek(fid, 0, 'eof');
    file_size_bytes = ftell(fid);
    fseek(fid, 0, 'bof');
    
    total_packets_in_file = floor(file_size_bytes / Bytes_Per_Packet_Arg);
    
    % Sliding Window Loop
    group_count = 0;
    
    for start_pkt = 0 : Step_Packets : (total_packets_in_file - Window_Packets)
        group_count = group_count + 1;
        
        % 1. Read Data Chunk
        offset_bytes = start_pkt * Bytes_Per_Packet_Arg;
        fseek(fid, offset_bytes, 'bof');
        
        solving_num = Window_Packets * 2; 
        
        adc_data_chunk_native = zeros(Fast_Time_Points, solving_num, Tx_Num*Rx_Num);
        read_success = true;
        
        for ii = 1:solving_num
            frame_data = fread(fid, Points_Per_Block, 'int16');
            if length(frame_data) < Points_Per_Block
                read_success = false; break; 
            end
            
            % Unpack
            frame_data_r = reshape(frame_data, Rx_Num, (Fast_Time_Points+60), Tx_Num, DS_Read);
            A_permuted = permute(frame_data_r(:, 1:Fast_Time_Points, :, :), [1 3 2, 4]); 
            A_permuted = permute(A_permuted, [3, 1, 2, 4]); 
            adc_data_r = reshape(A_permuted, [Fast_Time_Points, Rx_Num*Tx_Num, DS_Read]);
            
            adc_data_chunk_native(:, ii, :) = reshape(adc_data_r(:, :, 1), [Fast_Time_Points, 1, Rx_Num*Tx_Num]);
        end
        
        if ~read_success, break; end
        
        % Permute to [FastTime, Channels, SlowTime]
        adc_data = permute(adc_data_chunk_native, [1 3 2]); 
        
        % 2. Process Each Channel
        for k = 1:Total_Channels_To_Process
            % Extract Single Channel Data
            Raw_Single_Channel = squeeze(adc_data(:, k, :)); % [FastTime, SlowTime]
            
            % Range Compression
            Range_Profile_Complex = fft(Raw_Single_Channel, FFT_Points, 1);
            
            % ROI Selection
            RTM_ROI_Raw = Range_Profile_Complex(Min_Bin_Idx:Max_Bin_Idx, :);
            
            % Wall Clutter Removal via MTI
            RTM_MTI = RTM_ROI_Raw(:, 2:end) - RTM_ROI_Raw(:, 1:end-1);
            
            % Sum along range bins to get time series for DTM
            dtm_time_series = sum(RTM_MTI, 1);
            
            % STFT Calculation
            [S_stft, ~, ~] = stft(dtm_time_series, PRF_Effective, ...
                'Window', STFT_Window, 'OverlapLength', STFT_Overlap, 'FFTLength', N_fft_doppler);
            
            % Normalize Amplitude
            Mag_DTM = abs(S_stft);
            Norm_DTM = Mag_DTM / max(Mag_DTM(:));
            
            % 3. Save Data
            ch_root = sprintf('%s%d', Output_Root_Base, k);
            save_dir = fullfile(ch_root, name_no_ext);
            filename = fullfile(save_dir, sprintf('%s_Sample_%d.mat', name_no_ext, group_count));
            
            save(filename, 'Norm_DTM');
        end
        
        if mod(group_count, 50) == 0
            fprintf('    > Generated %d samples (x8 channels)...\n', group_count);
        end
    end
    
    fclose(fid);
    fprintf('  > Finished %s. Total Samples: %d.\n', current_file_name, group_count);
end

disp('Measured DTM Dataset Generation Complete.');