%% Script for Measured Complex-Valued RTD Dataset Generation
% Original Author: Longzhen Tang, Shisheng Guo, Qiang Jian, Guolong Cui, Lingjiang Kong, and Xiaobo Yang.
% Reproduced By: JoeyBG.
% Date: 2025-12-27.
% Affiliation: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description:
%   1. Reads raw measured radar data (.NOV) using the strict sliding window logic from Measured_RTMSet_Generator.
%   2. Extracts Channel 1 as the primary feature source.
%   3. Processes: Range Comp -> MTI -> Keystone Transform -> 3D Sliding Window FFT.
%   4. Saves 'RTD_Feature' as .mat files in 'Measured_RTDSet'.

%% Initialization
clear all;
close all;
clc;
disp('---------- © Author: JoeyBG © ----------');

%% Global Parameter Definitions
% --- File and Directory Settings ---
Data_Folder = "D:\JoeyBG_Research_Production\TWR_Identity_Threat\RW_Set\RW_Datas";
Output_Root = "Measured_RTDSet";

% Define Classes and Source Files
classes = struct(...
    'Name',      {'P1_Gun',      'P1_Nogun',    'P2_Gun',      'P2_Nogun', ...
                  'P3_Gun',      'P3_Nogun',    'P4_Gun',      'P4_Nogun'}, ...
    'FileName',  {'P1_Gun.NOV',  'P1_Nogun.NOV','P2_Gun.NOV',  'P2_Nogun.NOV', ...
                  'P3_Gun.NOV',  'P3_Nogun.NOV','P4_Gun.NOV',  'P4_Nogun.NOV'} ...
);

% --- Reading Parameters ---
Window_Packets = 100;                                                       % Packets per sample
Step_Packets = 12;                                                          % Overlap/Stride for data augmentation
Fast_Time_Points = 3940;                                                    
Rx_Num = 8;
Tx_Num = 8;
Header_Len = 60;
Bytes_Per_Int16 = 2;

% Block Calculation for fseek
DS_Read = 1; 
Points_Per_Block = Rx_Num * (Fast_Time_Points + Header_Len) * Tx_Num * DS_Read;
Bytes_Per_Block = Points_Per_Block * Bytes_Per_Int16;
Bytes_Per_Packet_Arg = Bytes_Per_Block * 2;                                 

% --- Radar Physics Parameters ---
c = 3e8;
fc = 2.5e9;                                                                 % Carrier Freq
B = 1.0e9;                                                                  % Bandwidth
PRT = 1e-3;                                                                 % Pulse Repetition Time
fs = 4e6;                                                                   % Sampling Rate
Process_DS_Ratio = 5;                                                       
Total_DS_Ratio = DS_Read * Process_DS_Ratio;
PRF_Effective = (1/PRT) / Total_DS_Ratio;

% --- Wall Compensation ---
Thickness_Wall = 0.12;    
Wall_Dieletric = 6;       
Wall_Compensation_Dist = Thickness_Wall * (sqrt(Wall_Dieletric) - 1);

% --- RTD Feature Generation Parameters ---
FFT_Points = 4096;                                                          % Range FFT
MTI_Step = 1;                                                               % MTI Step
Win_Length = 20;                                                            % Sliding Window Length
Win_Stride = 2;                                                             % Sliding Window Stride
N_Doppler_FFT = 64;                                                         % Doppler FFT size

% --- Channel Selection ---
Target_Channel = 1;                                                         % Using Channel 1 for Dataset

%% Dataset Structure Configuration
create_dataset_dirs(Output_Root, classes);

%% Main Processing Loop
fprintf('Starting Measured CV-RTD Dataset Generation...\n');

for c_idx = 1:length(classes)
    cls = classes(c_idx);
    full_path = fullfile(Data_Folder, cls.FileName);
    
    fprintf('\nProcessing Class: %s (Source: %s)\n', cls.Name, cls.FileName);
    
    if ~isfile(full_path)
        warning('File not found: %s. Skipping.', full_path);
        continue;
    end
    
    fid = fopen(full_path, 'r');
    if fid == -1
        warning('Cannot open file: %s', full_path);
        continue;
    end
    
    % Determine total packets
    fseek(fid, 0, 'eof');
    file_size_bytes = ftell(fid);
    total_packets_in_file = floor(file_size_bytes / Bytes_Per_Packet_Arg);
    
    sample_count = 0;
    
    % Sliding Window Loop
    for start_pkt = 0 : Step_Packets : (total_packets_in_file - Window_Packets)
        sample_count = sample_count + 1;
        
        % 1. Read Data Chunk
        offset_bytes = start_pkt * Bytes_Per_Packet_Arg;
        fseek(fid, offset_bytes, 'bof');
        
        solving_num = Window_Packets * 2; 
        
        adc_data_chunk_native = zeros(Fast_Time_Points, solving_num, Rx_Num*Tx_Num);
        read_success = true;
        
        for ii = 1:solving_num
            frame_data = fread(fid, Points_Per_Block, 'int16');
            if length(frame_data) < Points_Per_Block
                read_success = false; break; 
            end
            
            frame_data_r = reshape(frame_data, Rx_Num, (Fast_Time_Points+60), Tx_Num, DS_Read);
            A_permuted = permute(frame_data_r(:, 1:Fast_Time_Points, :, :), [1 3 2, 4]); 
            A_permuted = permute(A_permuted, [3, 1, 2, 4]); 
            adc_data_r = reshape(A_permuted, [Fast_Time_Points, Rx_Num*Tx_Num, DS_Read]);
            
            adc_data_chunk_native(:, ii, :) = reshape(adc_data_r(:, :, 1), [Fast_Time_Points, 1, Rx_Num*Tx_Num]);
        end
        
        if ~read_success, break; end
        
        % Permute to [FastTime, Channels, SlowTime]
        adc_data_all = permute(adc_data_chunk_native, [1 3 2]);
        
        % 2. Signal Processing
        
        % A. Extract Target Channel
        Raw_Ch = squeeze(adc_data_all(:, Target_Channel, :)); % [FastTime x SlowTime]
        
        % B. Range Compression
        Range_Profile = fft(Raw_Ch, FFT_Points, 1);
        Range_Profile = Range_Profile(1:FFT_Points/2, :);
        [Num_Range_Bins, Num_Pulses_MTI_Raw] = size(Range_Profile);
        
        % C. MTI Clutter Removal
        Range_Profile_MTI = Range_Profile(:, MTI_Step+1:end) - Range_Profile(:, 1:end-MTI_Step);
        Range_Profile_MTI = [Range_Profile_MTI, Range_Profile_MTI(:,end)]; % Pad
        [~, Num_Pulses_MTI] = size(Range_Profile_MTI);
        
        % D. Keystone Transform
        RT_Keystone = zeros(size(Range_Profile_MTI));
        slow_time_indices = 0:Num_Pulses_MTI-1;
        
        % Frequency Axis for Keystone
        Range_Resolution_Raw = c / (2 * B);
        freq_axis_beat = (0:Num_Range_Bins-1)' * (fs / FFT_Points);
        freq_axis_rf = fc + freq_axis_beat; 
        
        for r = 1:Num_Range_Bins
            f_curr = freq_axis_rf(r);
            scale_factor = f_curr / fc;
            t_new = slow_time_indices * scale_factor;
            
            % Interpolation
            RT_Keystone(r, :) = interp1(slow_time_indices, Range_Profile_MTI(r, :), t_new, 'linear', 0);
        end
        
        % E. 3D Complex-Valued RTD Construction
        num_windows = floor((Num_Pulses_MTI - Win_Length) / Win_Stride) + 1;
        RTD_Feature = zeros(Num_Range_Bins, N_Doppler_FFT, num_windows);
        
        Doppler_Window = hamming(Win_Length);
        
        for w = 1:num_windows
            idx_start = (w-1)*Win_Stride + 1;
            idx_end = idx_start + Win_Length - 1;
            
            segment = RT_Keystone(:, idx_start:idx_end);
            segment_win = segment .* Doppler_Window';
            
            % FFT along slow time -> Doppler
            RD_Map = fftshift(fft(segment_win, N_Doppler_FFT, 2), 2);
            
            RTD_Feature(:, :, w) = RD_Map;
        end
        
        % 3. Save Data
        filename = sprintf('Sample_%d.mat', sample_count);
        save_path = fullfile(Output_Root, cls.Name, filename);
        save(save_path, 'RTD_Feature');
        
        if mod(sample_count, 50) == 0
            fprintf('    Class %s: Generated %d samples...\n', cls.Name, sample_count);
        end
    end
    
    fclose(fid);
end

disp('Measured RTD Dataset Generation Complete.');

%% Helper Function
function create_dataset_dirs(root, classes)
    if ~exist(root, 'dir'), mkdir(root); end
    for c = 1:length(classes)
        subpath = fullfile(root, classes(c).Name);
        if ~exist(subpath, 'dir'), mkdir(subpath); end
    end
end