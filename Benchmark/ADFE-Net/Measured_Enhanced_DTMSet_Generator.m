%% Batch Processing Script: Measured Enhanced DTM Set Generator
% Original Author: Longzhen Tang, Shisheng Guo, Jiachen Li, Junda Zhu, Guolong Cui, Lingjiang Kong, and Xiaobo Yang.
% Reproduced By: JoeyBG.
% Date: 2025-12-29.
% Affiliate: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description:
%   1. Processes raw radar data (.NOV) from 'RW_Datas' folder in batches.
%   2. Uses a sliding window to slice data.
%   3. For each slice:
%      a. Generates Range Profile and Selects Best Channel.
%      b. Applies TPC and AC.
%      c. Generates DTMs via STFT summation.
%      d. Fuses TPC and AC spectrograms using Adaptive Weighted Fusion.
%   4. Saves the resulting Enhanced DTM as .mat files in 'Measured_Enhanced_DTMSet'.

%% Initialization
clear all; 
close all; 
clc; 
disp('---------- © Author: JoeyBG © ----------');

%% Parameter Definitions
% --- File and Directory Settings ---
Data_Folder = "D:\JoeyBG_Research_Production\TWR_Identity_Threat\RW_Set\RW_Datas";
Output_Root = "Measured_Enhanced_DTMSet";

File_List = [
    "P1_Nogun.NOV", "P1_Gun.NOV", ...
    "P2_Nogun.NOV", "P2_Gun.NOV", ...
    "P3_Nogun.NOV", "P3_Gun.NOV", ...
    "P4_Nogun.NOV", "P4_Gun.NOV"
];

% --- Reading Parameters ---
Window_Packets = 100;                                                       % 100 packets per sample
Step_Packets = 12;                                                          % Sliding step
Fast_Time_Points = 3940;                                                    
Rx_Num = 8;
Tx_Num = 8;
Header_Len = 60;
Bytes_Per_Int16 = 2;

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
PRF_Effective = (1/T) / Total_DS_Ratio;                                     % Effective PRF after downsampling

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

% --- Spectrogram & Fusion Parameters ---
MTI_Step = 1;                                                               % Only for Entropy Calc
STFT_Win_Size = 20;                                                         % Window size
STFT_Overlap = 18;                                                          % Overlap
N_fft_doppler = 256;                                                        % Doppler FFT size
Fusion_Bandwidth_Sigma = 10;                                                % Fusion Sigma

%% Main Processing Loop
fprintf('Starting Measured Enhanced DTM Generation...\n');

% Pre-calculate Fusion Weights
freq_indices = linspace(-PRF_Effective/2, PRF_Effective/2, N_fft_doppler)';
Weight_AC_Vec = exp(- (freq_indices.^2) / (2 * Fusion_Bandwidth_Sigma^2));
Weight_TPC_Vec = 1 - Weight_AC_Vec;
STFT_Window = hamming(STFT_Win_Size);

for f_idx = 1:length(File_List)
    current_file_name = File_List(f_idx);
    full_path = fullfile(Data_Folder, current_file_name);
    
    [~, name_no_ext, ~] = fileparts(current_file_name);
    
    % Create Class Directory
    save_dir = fullfile(Output_Root, name_no_ext);
    if ~exist(save_dir, 'dir'), mkdir(save_dir); end
    
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
        [~, Total_Channels, ~] = size(adc_data);
        
        % 2. RTM Generation & Channel Selection
        Range_Profile_Complex = fft(adc_data, FFT_Points, 1);
        RTM_ROI_Raw = Range_Profile_Complex(Min_Bin_Idx:Max_Bin_Idx, :, :);
        
        % Temporary MTI for Entropy Calculation
        RTM_MTI_Temp = RTM_ROI_Raw(:, :, MTI_Step+1:end) - RTM_ROI_Raw(:, :, 1:end-MTI_Step);
        
        entropies = zeros(1, Total_Channels);
        for ch = 1:Total_Channels
            temp_img = mat2gray(abs(squeeze(RTM_MTI_Temp(:, ch, :))));
            entropies(ch) = entropy(im2uint8(temp_img));
        end
        [~, Ref_Idx] = min(entropies);
        
        % Extract Best Channel RAW Data
        % Dimensions: [Range, Time]
        Best_RTM_Complex = squeeze(RTM_ROI_Raw(:, Ref_Idx, :));
        
        % 3. Clutter Suppression
        % TPC: Subtract consecutive pulses
        RTM_TPC = Best_RTM_Complex(:, 2:end) - Best_RTM_Complex(:, 1:end-1);
        
        % AC: Subtract mean
        Mean_Clutter = mean(Best_RTM_Complex, 2);
        RTM_AC = Best_RTM_Complex - Mean_Clutter;
        RTM_AC = RTM_AC(:, 1:end-1); % Align dimensions
        
        % 4. Enhanced Spectrogram Generation
        % STFT Function: [Range, Time] -> Transpose -> STFT -> Sum Range
        generate_DTM = @(RTM_Input) squeeze(sum(abs(stft(RTM_Input.', PRF_Effective, ...
            'Window', STFT_Window, 'OverlapLength', STFT_Overlap, 'FFTLength', N_fft_doppler)), 3));
        
        % Generate DTMs
        DTM_TPC = generate_DTM(RTM_TPC);
        DTM_AC  = generate_DTM(RTM_AC);
        
        % Normalize
        max_tpc = max(DTM_TPC(:)); if max_tpc == 0, max_tpc = 1; end
        max_ac  = max(DTM_AC(:));  if max_ac == 0, max_ac = 1; end
        
        DTM_TPC_Norm = DTM_TPC / max_tpc;
        DTM_AC_Norm  = DTM_AC / max_ac;
        
        % 5. Adaptive Fusion
        % Expand weights to match time dimension
        Num_Time_Bins = size(DTM_TPC_Norm, 2);
        Weight_AC_Map = repmat(Weight_AC_Vec, 1, Num_Time_Bins);
        Weight_TPC_Map = repmat(Weight_TPC_Vec, 1, Num_Time_Bins);
        
        % Fuse
        Enhanced_DTM = (Weight_TPC_Map .* DTM_TPC_Norm) + (Weight_AC_Map .* DTM_AC_Norm);
        Enhanced_DTM = Enhanced_DTM / max(Enhanced_DTM(:));
        
        % 6. Save Data
        % Save variable 'Enhanced_DTM' [Freq x Time]
        filename = fullfile(save_dir, sprintf('%s_Group_%d.mat', name_no_ext, group_count));
        save(filename, 'Enhanced_DTM');
        
        if mod(group_count, 20) == 0
            fprintf('    > Generated %d samples...\n', group_count);
        end
    end
    
    fclose(fid);
    fprintf('  > Finished %s. Total Samples: %d.\n', current_file_name, group_count);
end

disp('Dataset Generation Completes.');