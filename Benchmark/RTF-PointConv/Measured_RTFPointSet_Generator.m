%% Batch Processing Script: Measured RTF Point Set Generator
% Original Author: Hang Xu, Yong Li, Qingran Dong, Li Liu, Jingxia Li, Jianguo Zhang, and Bingjie Wang.
% Improved By: JoeyBG.
% Date: 2025-12-25.
% Affiliate: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description:
%   1. Processes raw radar data (.NOV) from 'RW_Datas' folder in batches.
%   2. Uses a sliding window to slice data.
%   3. For each slice:
%      a. Generates RTM and performs MTI.
%      b. Selects the best channel based on Minimum Entropy.
%      c. Constructs RTF Cube via STFT.
%      d. Performs Thresholding and Farthest Point Sampling (FPS).
%   4. Saves the resulting 1024x3 Point Cloud as .mat files in 'Measured_RTFPointSet'.

%% Initialization
clear all; 
close all; 
clc; 
disp('---------- © Author: JoeyBG © ----------');

%% Parameter Definitions
% --- File and Directory Settings ---
Data_Folder = "D:\JoeyBG_Research_Production\TWR_Identity_Threat\RW_Set\RW_Datas";
Output_Root = "Measured_RTFPointSet";

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
Range_Axis_Show = Range_Axis_Physical(Min_Bin_Idx:Max_Bin_Idx);

% --- RTF & Point Cloud Parameters ---
MTI_Step = 1;
FPS_Point_Count = 1024;
Intensity_Retention_Ratio = 0.05;                                           % Top 5% for real data
STFT_Win_Size = 20;                                                         % Smaller window for short duration
STFT_Overlap = 18;
N_fft_doppler = 256;

%% Main Processing Loop
fprintf('Starting Measured Point Cloud Generation...\n');

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
        
        % --- 1. Read Data Chunk ---
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
        
        % --- 2. RTM Generation & MTI ---
        Range_Profile_Complex = fft(adc_data, FFT_Points, 1);
        RTM_ROI_Raw = Range_Profile_Complex(Min_Bin_Idx:Max_Bin_Idx, :, :);
        % MTI
        RTM_MTI = RTM_ROI_Raw(:, :, MTI_Step+1:end) - RTM_ROI_Raw(:, :, 1:end-MTI_Step);
        [nRange, ~, Final_Frames] = size(RTM_MTI);
        
        % --- 3. Best Channel Selection ---
        entropies = zeros(1, Total_Channels);
        for ch = 1:Total_Channels
            temp_img = mat2gray(abs(squeeze(RTM_MTI(:, ch, :))));
            entropies(ch) = entropy(im2uint8(temp_img));
        end
        [~, Ref_Idx] = min(entropies);
        
        % Extract Best Channel Complex Data [Range, Time]
        Best_RTM_Complex = squeeze(RTM_MTI(:, Ref_Idx, :));
        
        % --- 4. RTF Cube Construction ---
        STFT_Window = hamming(STFT_Win_Size);
        % Dummy STFT for dimensions
        [~, f_vec, t_vec] = stft(zeros(1, Final_Frames), PRF_Effective, ...
            'Window', STFT_Window, 'OverlapLength', STFT_Overlap, 'FFTLength', N_fft_doppler);
        
        Num_Time_Bins = length(t_vec);
        Num_Freq_Bins = length(f_vec);
        RTF_Cube = zeros(nRange, Num_Time_Bins, Num_Freq_Bins);
        
        for r = 1:nRange
            time_series = Best_RTM_Complex(r, :);
            [S_stft, ~, ~] = stft(time_series, PRF_Effective, ...
                'Window', STFT_Window, 'OverlapLength', STFT_Overlap, 'FFTLength', N_fft_doppler);
            RTF_Cube(r, :, :) = abs(S_stft).'; % [Range, Time, Doppler]
        end
        
        RTF_Cube = RTF_Cube / max(RTF_Cube(:));
        
        % --- 5. Point Cloud Sampling with CFAR + FPS ---
        % Thresholding
        threshold_val = quantile(RTF_Cube(:), 1 - Intensity_Retention_Ratio);
        [r_idx, t_idx, d_idx] = ind2sub(size(RTF_Cube), find(RTF_Cube > threshold_val));
        
        % Coordinate Mapping
        P_range = Range_Axis_Show(r_idx); P_range = P_range(:);
        P_time = t_vec(t_idx);            P_time = P_time(:);
        P_doppler = f_vec(d_idx);         P_doppler = P_doppler(:);
        
        Candidate_Points = [P_range, P_time, P_doppler];
        Num_Candidates = size(Candidate_Points, 1);
        
        % FPS Processing
        if Num_Candidates <= FPS_Point_Count
             if Num_Candidates == 0
                 Final_Points = zeros(FPS_Point_Count, 3);
             else
                 % Simple random resampling if points are insufficient
                 indices = randi(Num_Candidates, FPS_Point_Count, 1);
                 Final_Points = Candidate_Points(indices, :);
             end
        else
            sampled_indices = zeros(FPS_Point_Count, 1);
            dists = inf(Num_Candidates, 1);
            
            % Normalize for distance calc
            C_min = min(Candidate_Points);
            C_max = max(Candidate_Points);
            C_Norm = (Candidate_Points - C_min) ./ (C_max - C_min + 1e-6);
            
            current_id = randi(Num_Candidates);
            
            for k = 1:FPS_Point_Count
                sampled_indices(k) = current_id;
                curr_pt_norm = C_Norm(current_id, :);
                d_new = sum((C_Norm - curr_pt_norm).^2, 2);
                dists = min(dists, d_new);
                [~, current_id] = max(dists);
            end
            Final_Points = Candidate_Points(sampled_indices, :);
        end
        
        % --- 6. Save Data ---
        Point_Cloud = Final_Points; % 1024x3
        filename = fullfile(save_dir, sprintf('%s_Group_%d.mat', name_no_ext, group_count));
        save(filename, 'Point_Cloud');
        
        if mod(group_count, 20) == 0
            fprintf('    > Generated %d samples...\n', group_count);
        end
    end
    
    fclose(fid);
    fprintf('  > Finished %s. Total Samples: %d.\n', current_file_name, group_count);
end

disp('Dataset Generation Completes.');