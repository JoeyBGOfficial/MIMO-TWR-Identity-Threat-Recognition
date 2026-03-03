%% Script for Measured MIMO RTM Dataset Generation
% Original Author: Renming Liu, Yan Tang, Shaoming Zhang, Yusheng Li, and Jianmei Wang.
% Reproduced By: JoeyBG.
% Date: 2025-12-25.
% Affiliation: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description:
%   1. Reads raw measured radar data (.NOV) from 'RW_Datas' using sliding window.
%   2. Extracts 4 Fixed Channels (1, 18, 36, 54) to create a synchronized MIMO dataset.
%   3. Processes: Wall Compensation -> FFT -> MTI -> Linear Magnitude -> Normalize.
%   4. Saves RTMs as 1024x1024 pure images in separate channel folders.

%% Initialization
clear all;
close all;
clc;
disp('---------- © Author: JoeyBG © ----------');

%% Global Parameter Definitions
% --- File and Directory Settings ---
Data_Folder = "D:\JoeyBG_Research_Production\TWR_Identity_Threat\RW_Set\RW_Datas"; % Folder containing .NOV files
Output_Roots = {'Measured_Channel1', 'Measured_Channel2', 'Measured_Channel3', 'Measured_Channel4'};

% Define Classes and Source Files
classes = struct(...
    'Name',      {'P1_Gun',      'P1_Nogun',    'P2_Gun',      'P2_Nogun', ...
                  'P3_Gun',      'P3_Nogun',    'P4_Gun',      'P4_Nogun'}, ...
    'FileName',  {'P1_Gun.NOV',  'P1_Nogun.NOV','P2_Gun.NOV',  'P2_Nogun.NOV', ...
                  'P3_Gun.NOV',  'P3_Nogun.NOV','P4_Gun.NOV',  'P4_Nogun.NOV'} ...
);

% --- Reading Parameters ---
Window_Packets = 100;                                                       % Frames per sample
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
Bytes_Per_Packet_Arg = Bytes_Per_Block * 2;                                 % Packet size in bytes

% --- Radar Physics Parameters ---
c = 3e8;
f0 = 2.5e9; 
B = 1.0e9; 
PRT = 1e-3; 
fs = 4e6; 

Process_DS_Ratio = 1;                                                       % Assuming no extra DS during processing
Total_DS_Ratio = DS_Read * Process_DS_Ratio;

% --- Wall Compensation ---
Thickness_Wall = 0.12;    
Wall_Dieletric = 6;       
Wall_Compensation_Dist = Thickness_Wall * (sqrt(Wall_Dieletric) - 1);

% --- Range Axis and ROI ---
FFT_Points = 4096;
Range_Resolution = c / (2 * B);
Range_Axis_Raw = ((0:FFT_Points/2-1) * Range_Resolution) / 2;
Range_Axis_Physical = Range_Axis_Raw - Wall_Compensation_Dist;

Min_Range_ROI = 0;                                                          % ROI Start (m)
Max_Range_ROI = 6.0;                                                        % ROI End (m)
[~, Min_Bin_Idx] = min(abs(Range_Axis_Physical - Min_Range_ROI));
[~, Max_Bin_Idx] = min(abs(Range_Axis_Physical - Max_Range_ROI));

% --- Channel Selection ---
Selected_Channels = [1, 18, 36, 54];                                        % Fixed 4 Channels
N_Channels_Out = 4;

% --- Image Generation Parameters ---
MTI_Step = 1;
Target_Img_Size = [1024, 1024];
Colormap_Name = 'jet';

%% Dataset Structure Configuration
create_dataset_dirs(Output_Roots, classes);

%% Main Processing Loop
fprintf('Starting Measured MIMO RTM Dataset Generation...\n');

for c_idx = 1:length(classes)
    cls = classes(c_idx);
    full_path = fullfile(Data_Folder, cls.FileName);
    
    fprintf('\nProcessing Class: %s (Source: %s)\n', cls.Name, cls.FileName);
    
    % Check file
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
    
    % Sliding Window Loop
    sample_count = 0;
    
    % Loop until end of file
    for start_pkt = 0 : Step_Packets : (total_packets_in_file - Window_Packets)
        sample_count = sample_count + 1;
        
        % --- 1. Read Data Chunk ---
        offset_bytes = start_pkt * Bytes_Per_Packet_Arg;
        fseek(fid, offset_bytes, 'bof');
        
        % We need to read Window_Packets frames
        % Each frame contains 64 channels
        solving_num = Window_Packets * 2; % Factor of 2 from read logic
        
        % Buffer for [FastTime, Frames, Channels]
        adc_data_chunk_native = zeros(Fast_Time_Points, solving_num, Rx_Num*Tx_Num);
        read_success = true;
        
        for ii = 1:solving_num
            frame_data = fread(fid, Points_Per_Block, 'int16');
            if length(frame_data) < Points_Per_Block
                read_success = false; break; 
            end
            
            % Unpack Data Structure
            frame_data_r = reshape(frame_data, Rx_Num, (Fast_Time_Points+60), Tx_Num, DS_Read);
            A_permuted = permute(frame_data_r(:, 1:Fast_Time_Points, :, :), [1 3 2, 4]); 
            A_permuted = permute(A_permuted, [3, 1, 2, 4]); 
            adc_data_r = reshape(A_permuted, [Fast_Time_Points, Rx_Num*Tx_Num, DS_Read]);
            
            % Store
            adc_data_chunk_native(:, ii, :) = reshape(adc_data_r(:, :, 1), [Fast_Time_Points, 1, Rx_Num*Tx_Num]);
        end
        
        if ~read_success, break; end
        
        % Permute to Standard [FastTime, Channels, SlowTime]
        adc_data_all = permute(adc_data_chunk_native, [1 3 2]);
        
        % --- 2. Process Each Selected Channel ---
        for ch_out_idx = 1:N_Channels_Out
            phy_ch = Selected_Channels(ch_out_idx);
            
            % Extract Channel Data: [FastTime, SlowTime]
            Raw_Ch = squeeze(adc_data_all(:, phy_ch, :));
            
            % A. Pulse Compression
            Range_Profile = fft(Raw_Ch, FFT_Points, 1);
            
            % B. Cut ROI
            RTM_ROI = Range_Profile(Min_Bin_Idx:Max_Bin_Idx, :);
            
            % C. MTI Clutter Removal
            RTM_MTI = RTM_ROI(:, MTI_Step+1:end) - RTM_ROI(:, 1:end-MTI_Step);
            
            % D. Linear Magnitude
            RTM_Mag = abs(RTM_MTI);
            
            % E. Normalize
            min_val = min(RTM_Mag(:));
            max_val = max(RTM_Mag(:));
            if max_val > min_val
                RTM_Norm = (RTM_Mag - min_val) / (max_val - min_val);
            else
                RTM_Norm = RTM_Mag;
            end
            
            % --- 3. Save Image ---
            % Resize to 1024x1024
            Img_Resized = imresize(RTM_Norm, Target_Img_Size);
            
            % Apply Jet Colormap
            RGB_Img = ind2rgb(gray2ind(Img_Resized, 256), jet(256));
            
            % Save
            filename = sprintf('Sample_%d.png', sample_count);
            save_path = fullfile(Output_Roots{ch_out_idx}, cls.Name, filename);
            imwrite(RGB_Img, save_path);
        end
        
        if mod(sample_count, 50) == 0
            fprintf('    Class %s: Generated %d samples...\n', cls.Name, sample_count);
        end
    end
    
    fclose(fid);
end

disp('Measured MIMO RTM Dataset Generation Completes.');

%% Helper Function
function create_dataset_dirs(root_paths, classes)
    for r = 1:length(root_paths)
        root = root_paths{r};
        if ~exist(root, 'dir'), mkdir(root); end
        for c = 1:length(classes)
            subpath = fullfile(root, classes(c).Name);
            if ~exist(subpath, 'dir'), mkdir(subpath); end
        end
    end
end