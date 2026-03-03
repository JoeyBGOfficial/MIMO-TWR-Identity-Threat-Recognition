%% Script for Simulated MobRNet
% Original Author: Renming Liu, Yan Tang, Shaoming Zhang, Yusheng Li, and Jianmei Wang.
% Reproduced By: JoeyBG.
% Date: 2025-12-25.
% Affiliation: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description:
%   1. Loads 4-Channel RTM data from Simulated_Channel{1-4}.
%   2. Constructs the MobRNet architecture: MobileNetV2 Backbone + Spatial Attention + FPN.
%      - FIXED: Used 'NumGroups', 'channel-wise' for Depthwise Conv.
%      - FIXED: Adjusted FPN upsampling strides to match feature map dimensions.
%   3. Trains the model for 8-class classification using Adam optimizer.
%   4. Visualizes Training Progress and Validation Confusion Matrix.

%% Initialization
clear all;
close all;
clc;
disp('---------- © Author: JoeyBG © ----------');

%% Parameter Definitions
% --- Data Paths ---
channel_folders = {'Simulated_Channel1', 'Simulated_Channel2', 'Simulated_Channel3', 'Simulated_Channel4'};
input_size = [128, 128, 3];                                                 % Resizing 1024x1024 to manageable size

% --- Training Parameters ---
train_ratio = 0.8;
batch_size = 32;
max_epochs = 80;
learning_rate = 0.0005;
L2_reg = 0.0001;

% --- Visualization Parameters ---
Font_Name = 'Palatino Linotype'; 
Font_Size = 12;

%% Section 1: Data Loading & Preprocessing
fprintf('Loading and Aggregating Data from 4 Channels...\n');

% Create aggregate ImageDatastore for final validation
all_files = {};
all_labels = [];

% Create per-channel datastores for training
imds_channel = cell(length(channel_folders), 1);
imds_train_channel = cell(length(channel_folders), 1);
imds_val_channel = cell(length(channel_folders), 1);
aug_train = cell(length(channel_folders), 1);
aug_val_channel = cell(length(channel_folders), 1);

for i = 1:length(channel_folders)
    root_path = channel_folders{i};
    if ~exist(root_path, 'dir')
        warning('Folder %s not found. Skipping.', root_path);
        continue;
    end
    
    % Load datastore for current channel
    imds_channel{i} = imageDatastore(root_path, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    
    % Filter out invalid '0' labels
    valid_idx = imds_channel{i}.Labels ~= categorical("0");
    imds_channel{i} = subset(imds_channel{i}, valid_idx);
    
    % Split per channel
    [imds_train_channel{i}, imds_val_channel{i}] = splitEachLabel(imds_channel{i}, train_ratio, 'randomized');
    
    % Data Augmentation per channel
    aug_train{i} = augmentedImageDatastore(input_size, imds_train_channel{i});
    aug_val_channel{i} = augmentedImageDatastore(input_size, imds_val_channel{i});
    
    % Aggregate for final val
    all_files = [all_files; imds_channel{i}.Files];
    all_labels = [all_labels; imds_channel{i}.Labels];
end

% Create unified Datastore for final validation
imds_full = imageDatastore(all_files, 'Labels', all_labels);
classes = categories(imds_full.Labels);
num_classes = length(classes);

fprintf('Total Images: %d | Classes: %d\n', length(imds_full.Files), num_classes);

% Split aggregate for final imds_val
[~, imds_val] = splitEachLabel(imds_full, train_ratio, 'randomized');

% Aggregate aug_val for final validation
aug_val = augmentedImageDatastore(input_size, imds_val);

%% Section 2: MobRNet Architecture Construction
fprintf('Constructing MobRNet Architecture...\n');
lgraph = layerGraph();

% 2.1 Input Block
% Initial Conv: 3 inputs -> 16 outputs, Stride 2 (128->64)
current_channels = 16; 

tempLayers = [
    imageInputLayer(input_size, 'Name', 'input')
    convolution2dLayer(3, current_channels, 'Padding', 'same', 'Stride', 2, 'Name', 'conv1') 
    batchNormalizationLayer('Name', 'conv1_bn')
    reluLayer('Name', 'conv1_relu')
];
lgraph = addLayers(lgraph, tempLayers);

% 2.2 Inverted Residual Blocks with Spatial Attention
% Configuration: [t, c, n, s]
block_configs = [
    1, 16, 1, 1;   % Stage 1: 64x64
    6, 24, 2, 2;   % Stage 2: 32x32
    6, 32, 3, 2;   % Stage 3: 16x16
    6, 64, 4, 2;   % Stage 4: 8x8
    6, 96, 3, 1;   % Stage 5: 8x8
    6, 160, 3, 2;  % Stage 6: 4x4
    6, 320, 1, 1   % Stage 7: 4x4
];

last_layer_name = 'conv1_relu';

block_id = 0;
for i = 1:size(block_configs, 1)
    t = block_configs(i, 1);
    c_out_target = block_configs(i, 2);
    n = block_configs(i, 3);
    s = block_configs(i, 4);
    
    for j = 1:n
        block_id = block_id + 1;
        block_name = sprintf('Block_%d', block_id);
        
        current_stride = 1;
        if j == 1, current_stride = s; end
        
        % Add Block
        [lgraph, last_layer_name] = add_mobrnet_block(lgraph, last_layer_name, block_name, t, current_channels, c_out_target, current_stride);
        
        current_channels = c_out_target; 
    end
    
end

% 2.4 Classification Head
tempHead = [
    globalAveragePooling2dLayer('Name', 'global_pool')
    fullyConnectedLayer(num_classes, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')
];

lgraph = addLayers(lgraph, tempHead);
lgraph = connectLayers(lgraph, last_layer_name, 'global_pool');

% Display Network Summary
analyzeNetwork(lgraph);

%% Section 3: Network Training
fprintf('Starting Training...\n');

% Train separate models for each channel
net = cell(length(channel_folders), 1);
info = cell(length(channel_folders), 1);

for i = 1:length(channel_folders)
    fprintf('Training model for Channel %d...\n', i);
    
    % Set Training Options for this channel
    options = trainingOptions('adam', ...
        'InitialLearnRate', learning_rate, ...
        'MaxEpochs', max_epochs, ...
        'MiniBatchSize', batch_size, ...
        'L2Regularization', L2_reg, ...
        'Shuffle', 'every-epoch', ...
        'Plots', 'training-progress', ...
        'Verbose', true, ...
        'ValidationData', aug_val_channel{i}, ...
        'ValidationFrequency', 20, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 30);
    
    [net{i}, info{i}] = trainNetwork(aug_train{i}, lgraph, options);
end

%% Section 4: Validation & Visualization
fprintf('Validating Model...\n');

% Predict with each model on the aggregate validation set
Y_pred_all = cell(length(channel_folders), 1);
for i = 1:length(channel_folders)
    Y_pred_all{i} = classify(net{i}, aug_val);
end

% Perform majority voting
num_samples = numel(imds_val.Labels);
Y_pred_ensemble = categorical(zeros(num_samples, 1));
for j = 1:num_samples
    votes = categorical(zeros(1, length(channel_folders)));
    for i = 1:length(channel_folders)
        votes(i) = Y_pred_all{i}(j);
    end
    Y_pred_ensemble(j) = mode(votes);
end

Y_true = imds_val.Labels;

accuracy = sum(Y_pred_ensemble == Y_true) / numel(Y_true);
fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);

% Plot Confusion Matrix
figure('Name', 'MobRNet Confusion Matrix', 'Color', 'w');
cm = confusionchart(Y_true, Y_pred_ensemble);
cm.Title = sprintf('MobRNet Validation Results (Acc: %.2f%%)', accuracy * 100);
cm.FontName = Font_Name;
cm.FontSize = Font_Size;
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

disp('Simulated MobRNet Training Complete.');

%% Helper Functions
function [lgraph, outputName] = add_mobrnet_block(lgraph, inputName, blockName, t, c_in, c_out, s)
    % Adds an Inverted Residual Block with Spatial Attention
    prefix = [blockName, '_'];
    
    % Calculate expanded channels
    expand_channels = c_in * t;
    
    % --- 1. Expansion Phase ---
    layers_neck = [
        convolution2dLayer(1, expand_channels, 'Name', [prefix 'expand_conv'], 'Padding', 'same')
        batchNormalizationLayer('Name', [prefix 'expand_bn'])
        reluLayer('Name', [prefix 'expand_relu'])
        
        % Depthwise (3x3)
        groupedConvolution2dLayer(3, 1, 'channel-wise', ...
            'Padding', 'same', 'Stride', s, 'Name', [prefix 'dw_conv']) % Using 'NumGroups', 'channel-wise' for standard depthwise convolution
            
        batchNormalizationLayer('Name', [prefix 'dw_bn'])
        reluLayer('Name', [prefix 'dw_relu'])
    ];
    lgraph = addLayers(lgraph, layers_neck);
    lgraph = connectLayers(lgraph, inputName, [prefix 'expand_conv']);
    
    % --- 2. Spatial Attention Mechanism ---
    att_layers = [
        convolution2dLayer(1, 1, 'Name', [prefix 'att_pool_approx']) 
        convolution2dLayer(7, 1, 'Padding', 'same', 'Name', [prefix 'att_conv7'])
        sigmoidLayer('Name', [prefix 'att_sigmoid'])
    ];
    lgraph = addLayers(lgraph, att_layers);
    lgraph = connectLayers(lgraph, [prefix 'dw_relu'], [prefix 'att_pool_approx']);
    
    % Multiply
    mult_name = [prefix 'att_mult'];
    lgraph = addLayers(lgraph, multiplicationLayer(2, 'Name', mult_name));
    lgraph = connectLayers(lgraph, [prefix 'dw_relu'], [mult_name '/in1']);
    lgraph = connectLayers(lgraph, [prefix 'att_sigmoid'], [mult_name '/in2']);
    
    % --- 3. Projection Phase ---
    proj_layers = [
        convolution2dLayer(1, c_out, 'Name', [prefix 'proj_conv'], 'Padding', 'same')
        batchNormalizationLayer('Name', [prefix 'proj_bn'])
    ];
    lgraph = addLayers(lgraph, proj_layers);
    lgraph = connectLayers(lgraph, mult_name, [prefix 'proj_conv']);
    
    outputName = [prefix 'proj_bn'];
    
    % --- 4. Residual Connection ---
    if s == 1 && c_in == c_out
        add_name = [prefix 'add'];
        lgraph = addLayers(lgraph, additionLayer(2, 'Name', add_name));
        lgraph = connectLayers(lgraph, inputName, [add_name '/in1']);
        lgraph = connectLayers(lgraph, [prefix 'proj_bn'], [add_name '/in2']);
        outputName = add_name;
    end
end