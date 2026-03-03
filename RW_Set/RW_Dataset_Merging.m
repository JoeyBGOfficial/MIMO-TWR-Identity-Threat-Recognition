%% Batch Processing Script: Dataset Merger for PSNR + TRGS
% Original Author: JoeyBG.
% Modified By: JoeyBG.
% Date: 2025-12-05.
% Affiliate: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Introduction:
%   This script merges two distinct datasets (PSNR-based and TRGS-based) into a single unified dataset.
%   Methodology: 
%       1. Iterates through all category subfolders.
%       2. Processes 'PSNR' images first, assigning them indices 1 to N.
%       3. Processes 'TRGS' images next, assigning them indices N+1 to M.
%       4. Ensures RTM, DTM, and RDM images maintain strict index synchronization during renaming.
%   Output:
%       Unified folders 'RW_RTM_Set', 'RW_DTM_Set', and 'RW_RDM_Set' containing sequentially numbered PNGs.
%
% Source Directories:
%   RW_RTM_Set_PSNR, RW_DTM_Set_PSNR, RW_RDM_Set_PSNR
%   RW_RTM_Set_TRGS, RW_DTM_Set_TRGS, RW_RDM_Set_TRGS

%% Initialization of MATLAB Script
clear all; 
close all; 
clc;
disp('---------- © Author: JoeyBG © ----------');

%% Parameter Definitions
% Input directory settings
Source_Dirs.RTM_PSNR = "RW_RTM_Set_PSNR";
Source_Dirs.DTM_PSNR = "RW_DTM_Set_PSNR";
Source_Dirs.RDM_PSNR = "RW_RDM_Set_PSNR";

Source_Dirs.RTM_TRGS = "RW_RTM_Set_TRGS";
Source_Dirs.DTM_TRGS = "RW_DTM_Set_TRGS";
Source_Dirs.RDM_TRGS = "RW_RDM_Set_TRGS";

% Output directory settings
Output_Dirs.RTM = "RW_RTM_Set";
Output_Dirs.DTM = "RW_DTM_Set";
Output_Dirs.RDM = "RW_RDM_Set";

% Create output roots
if ~exist(Output_Dirs.RTM, 'dir'), mkdir(Output_Dirs.RTM); end
if ~exist(Output_Dirs.DTM, 'dir'), mkdir(Output_Dirs.DTM); end
if ~exist(Output_Dirs.RDM, 'dir'), mkdir(Output_Dirs.RDM); end

% Scan for category subfolders
Dir_Struct = dir(Source_Dirs.RTM_PSNR);
Dir_Mask = [Dir_Struct.isdir] & ~ismember({Dir_Struct.name}, {'.', '..'}); 
Category_List = {Dir_Struct(Dir_Mask).name};

%% Main Processing Loop
fprintf('Starting Dataset Merger Process (RTM + DTM + RDM)...\n');

for c_idx = 1:length(Category_List)
    Current_Category = Category_List{c_idx};
    fprintf('Processing Category (%d/%d): %s ...\n', c_idx, length(Category_List), Current_Category);
    
    % Create category subdirectories in output
    Out_Sub_RTM = fullfile(Output_Dirs.RTM, Current_Category);
    Out_Sub_DTM = fullfile(Output_Dirs.DTM, Current_Category);
    Out_Sub_RDM = fullfile(Output_Dirs.RDM, Current_Category);
    
    if ~exist(Out_Sub_RTM, 'dir'), mkdir(Out_Sub_RTM); end
    if ~exist(Out_Sub_DTM, 'dir'), mkdir(Out_Sub_DTM); end
    if ~exist(Out_Sub_RDM, 'dir'), mkdir(Out_Sub_RDM); end
    
    % Initialize global counter for this category
    Global_Index = 1; % This ensures PSNR 1..N and TRGS N+1..M are continuous
    
    % ---------------------------------------------------------------------
    % 1. Process PSNR dataset
    % ---------------------------------------------------------------------
    Src_RTM_PSNR = fullfile(Source_Dirs.RTM_PSNR, Current_Category);
    Src_DTM_PSNR = fullfile(Source_Dirs.DTM_PSNR, Current_Category);
    Src_RDM_PSNR = fullfile(Source_Dirs.RDM_PSNR, Current_Category);
    
    [Global_Index] = Process_And_Copy_Triplet(Src_RTM_PSNR, Src_DTM_PSNR, Src_RDM_PSNR, ...
        Out_Sub_RTM, Out_Sub_DTM, Out_Sub_RDM, Current_Category, Global_Index);
        
    % ---------------------------------------------------------------------
    % 2. Process TRGS dataset
    % ---------------------------------------------------------------------
    Src_RTM_TRGS = fullfile(Source_Dirs.RTM_TRGS, Current_Category);
    Src_DTM_TRGS = fullfile(Source_Dirs.DTM_TRGS, Current_Category); 
    Src_RDM_TRGS = fullfile(Source_Dirs.RDM_TRGS, Current_Category);
    
    [Global_Index] = Process_And_Copy_Triplet(Src_RTM_TRGS, Src_DTM_TRGS, Src_RDM_TRGS, ...
        Out_Sub_RTM, Out_Sub_DTM, Out_Sub_RDM, Current_Category, Global_Index);
        
    fprintf('  > Category %s finished. Total images merged: %d triplets.\n', Current_Category, Global_Index - 1);
end

fprintf('All datasets merged successfully!\n');

%% Helper Function: Process and Copy Triplets
function [Next_Index] = Process_And_Copy_Triplet(Src_Path_RTM, Src_Path_DTM, Src_Path_RDM, ...
                                                 Dst_Path_RTM, Dst_Path_DTM, Dst_Path_RDM, ...
                                                 Cat_Name, Start_Index)
    % Inputs:
    %   Src_Path_*: Source folders containing original images
    %   Dst_Path_*: Destination folders for merged dataset
    %   Cat_Name: Current category name for naming convention
    %   Start_Index: Starting number for renaming
    %
    % Outputs:
    %   Next_Index: The next available index after processing this batch
    
    % Get and sort files naturally
    Files_RTM = Get_Sorted_Files(Src_Path_RTM);
    Files_DTM = Get_Sorted_Files(Src_Path_DTM);
    Files_RDM = Get_Sorted_Files(Src_Path_RDM);
    
    % Validation
    Num_RTM = length(Files_RTM);
    Num_DTM = length(Files_DTM);
    Num_RDM = length(Files_RDM);
    
    if (Num_RTM ~= Num_DTM) || (Num_RTM ~= Num_RDM)
        warning('Count mismatch in %s: RTM=%d, DTM=%d, RDM=%d. Using RTM count as baseline.', ...
            Cat_Name, Num_RTM, Num_DTM, Num_RDM);
    end
    
    Current_Idx = Start_Index;
    
    % Batch copy loop
    for i = 1:Num_RTM
        Src_Name_RTM = Files_RTM(i).name;
        
        % Synchronization logic:
        %   We assume files correspond by sorted index.
        if i <= Num_DTM
            Src_Name_DTM = Files_DTM(i).name;
        else
            warning('Missing DTM file for %s. Skipping.', Src_Name_RTM);
            continue;
        end
        
        if i <= Num_RDM
            Src_Name_RDM = Files_RDM(i).name;
        else
            warning('Missing RDM file for %s. Skipping.', Src_Name_RTM);
            continue;
        end
        
        % Construct full source paths
        Full_Src_RTM = fullfile(Src_Path_RTM, Src_Name_RTM);
        Full_Src_DTM = fullfile(Src_Path_DTM, Src_Name_DTM);
        Full_Src_RDM = fullfile(Src_Path_RDM, Src_Name_RDM);
        
        % Extract Extension
        [~, ~, Ext] = fileparts(Src_Name_RTM);
        
        % Generate new unified name
        New_Name = sprintf('%s_Group_%d%s', Cat_Name, Current_Idx, Ext); % Format: Category_Group_Index.png        
        
        Full_Dst_RTM = fullfile(Dst_Path_RTM, New_Name);
        Full_Dst_DTM = fullfile(Dst_Path_DTM, New_Name);
        Full_Dst_RDM = fullfile(Dst_Path_RDM, New_Name);
        
        % Perform copy
        copyfile(Full_Src_RTM, Full_Dst_RTM);
        copyfile(Full_Src_DTM, Full_Dst_DTM);
        copyfile(Full_Src_RDM, Full_Dst_RDM);
        
        Current_Idx = Current_Idx + 1;
    end
    
    Next_Index = Current_Idx;
end

%% Helper Function: Natural Sorting of Files
function File_List = Get_Sorted_Files(Folder_Path)
    % Inputs:
    %   Folder_Path: Path to the directory to scan
    %
    % Outputs:
    %   File_List: Struct array of files, sorted by the number in 'Group_X'
    
    if ~exist(Folder_Path, 'dir')
        warning('Directory not found: %s', Folder_Path);
        File_List = [];
        return;
    end
    
    Raw_List = dir(fullfile(Folder_Path, '*.png')); 
    
    if isempty(Raw_List)
        File_List = [];
        return;
    end
    
    File_Names = {Raw_List.name};
    Numbers = zeros(1, length(File_Names));
    
    % Extract numbers using regular expressions
    for i = 1:length(File_Names)
        % Pattern matches 'Group_' followed by digits
        Tokens = regexp(File_Names{i}, 'Group_(\d+)', 'tokens');
        if ~isempty(Tokens)
            Numbers(i) = str2double(Tokens{1}{1});
        else
            Numbers(i) = 0; % Fallback
        end
    end
    
    % Sort based on extracted numbers
    [~, Sort_Idx] = sort(Numbers);
    File_List = Raw_List(Sort_Idx);
end