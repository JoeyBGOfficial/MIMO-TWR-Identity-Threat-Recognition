%% Script for TWR HAR with Complex-Valued RTD and ConvGRU
% Original Author: Longzhen Tang, Shisheng Guo, Qiang Jian, Guolong Cui, Lingjiang Kong, and Xiaobo Yang.
% Reproduced By: JoeyBG.
% Date: 2025-12-27.
% Affiliation: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description:
%   1. Loads Complex-Valued 3D RTD datasets from 'Measured_RTDSet'.
%   2. Preprocesses data: Splits Real/Imag parts into 2 channels for network input.
%   3. Constructs a Spatiotemporal Network (CNN-GRU) to reproduce ConvGRU.
%   4. Trains using Adam optimizer with 8:2 Split.
%   5. Outputs Training Plots and Validation Confusion Matrix.

%% Initialization
clear all;
close all;
clc;
disp('---------- © Author: JoeyBG © ----------');

%% Parameter Definitions
% --- Path Parameters ---
data_root = 'Measured_RTDSet';
dir_info = dir(fullfile(data_root, 'P*'));
class_folders = {dir_info.name};
num_classes = length(class_folders);

% --- Training Parameters ---
train_ratio = 0.8;
batch_size = 32;                                                            % Smaller batch size for sequence data memory management
max_epochs = 80;
learning_rate = 0.0005;
L2_regularization = 0.0005;

% --- Visualization Parameters ---
Font_Name = 'Palatino Linotype'; 
Font_Size_Basis = 12;
Font_Size_Title = 14;

%% Section 1. Data Loading and Preprocessing
fprintf('Loading Complex-Valued RTD Dataset...\n');

X_seq = {};
Y_labels = [];
sample_counts = [];

% Loop through class folders
for i = 1:num_classes
    cls_name = class_folders{i};
    cls_path = fullfile(data_root, cls_name);
    files = dir(fullfile(cls_path, '*.mat'));
    
    if isempty(files)
        continue;
    end
    
    fprintf('  Loading Class %s: %d samples\n', cls_name, length(files));
    
    for k = 1:length(files)
        % Load RTD_Feature [Range x Doppler x Time]
        data_struct = load(fullfile(cls_path, files(k).name));
        rtd_complex = data_struct.RTD_Feature;
        
        % Data Preprocessing for Dual-Channel Input
        % Channel 1: Real Part, Channel 2: Imaginary Part
        rtd_real = real(rtd_complex);
        rtd_imag = imag(rtd_complex);
        
        % Stack into [Height x Width x Channels x Time]
        % MATLAB Sequence input expects each cell to be [H x W x C x T] usually,
        %   or [H x W x C] for each timestep.
        % Here we format for 'sequenceInputLayer': Cell array of [H x W x C x T]
        rtd_combined = cat(3, rtd_real, rtd_imag);
        
        % Add to dataset
        X_seq{end+1, 1} = rtd_combined; 
        Y_labels = [Y_labels; categorical(string(cls_name))];
    end
end

% Check Input Dimensions
sample_dim = size(X_seq{1});
input_size = sample_dim(1:3); % [Range, Doppler, 2]
fprintf('  Input Dimensions: %d Range, %d Doppler, %d Channels\n', ...
    input_size(1), input_size(2), input_size(3));

%% Section 2. Dataset Splitting
num_samples = length(Y_labels);
rand_idx = randperm(num_samples);
num_train = floor(train_ratio * num_samples);

idx_train = rand_idx(1:num_train);
idx_val = rand_idx(num_train+1:end);

X_train = X_seq(idx_train);
Y_train = Y_labels(idx_train);
X_val = X_seq(idx_val);
Y_val = Y_labels(idx_val);

fprintf('Dataset Split: %d Training, %d Validation.\n', length(Y_train), length(Y_val));

%% Section 3. ConvGRU Network Architecture Construction
% Note: In MATLAB, a "ConvGRU" that processes spatial features over time 
%   is implemented using a Sequence Folding Layer -> CNN Layers -> Sequence Unfolding -> GRU.
% This effectively applies the Convolutional kernels to every timestep and then aggregates temporally.
lgraph = layerGraph();

% 3.1 Input Layer
input_layer = sequenceInputLayer(input_size, ...
    'Name', 'Input', ...
    'Normalization', 'zscore'); % Z-score normalization crucial for RF data
lgraph = addLayers(lgraph, input_layer);

% 3.2 Sequence Folding
fold_layer = sequenceFoldingLayer('Name', 'Fold');
lgraph = addLayers(lgraph, fold_layer);
lgraph = connectLayers(lgraph, 'Input', 'Fold/in');

% 3.3 Spatial Feature Extraction
cnn_layers = [
    % Block 1
    convolution2dLayer([3, 3], 16, 'Padding', 'same', 'Name', 'Conv1')
    batchNormalizationLayer('Name', 'BN1')
    reluLayer('Name', 'ReLU1')
    maxPooling2dLayer([2, 2], 'Stride', 2, 'Name', 'Pool1') % Reduce spatial dim
    
    % Block 2
    convolution2dLayer([3, 3], 32, 'Padding', 'same', 'Name', 'Conv2')
    batchNormalizationLayer('Name', 'BN2')
    reluLayer('Name', 'ReLU2')
    maxPooling2dLayer([2, 2], 'Stride', 2, 'Name', 'Pool2')
    
    % Block 3
    convolution2dLayer([3, 3], 64, 'Padding', 'same', 'Name', 'Conv3')
    batchNormalizationLayer('Name', 'BN3')
    reluLayer('Name', 'ReLU3')
];
lgraph = addLayers(lgraph, cnn_layers);
lgraph = connectLayers(lgraph, 'Fold/out', 'Conv1');

% 3.4 Sequence Unfolding
unfold_layer = sequenceUnfoldingLayer('Name', 'Unfold');
lgraph = addLayers(lgraph, unfold_layer);
lgraph = connectLayers(lgraph, 'ReLU3', 'Unfold/in');
lgraph = connectLayers(lgraph, 'Fold/miniBatchSize', 'Unfold/miniBatchSize');

% 3.5 Flattening
flatten_layer = flattenLayer('Name', 'Flatten');
lgraph = addLayers(lgraph, flatten_layer);
lgraph = connectLayers(lgraph, 'Unfold/out', 'Flatten');

% 3.6 Temporal Processing (GRU)
gru_block = [
    gruLayer(128, 'OutputMode', 'last', 'Name', 'GRU') % 128 Hidden Units
    dropoutLayer(0.5, 'Name', 'Drop')
];
lgraph = addLayers(lgraph, gru_block);
lgraph = connectLayers(lgraph, 'Flatten', 'GRU');

% 3.7 Classification Head
classifier_block = [
    fullyConnectedLayer(num_classes, 'Name', 'FC')
    softmaxLayer('Name', 'Softmax')
    classificationLayer('Name', 'Output')
];
lgraph = addLayers(lgraph, classifier_block);
lgraph = connectLayers(lgraph, 'Drop', 'FC');

% Analyze Network
analyzeNetwork(lgraph);

%% Section 4. Network Training
fprintf('Starting Network Training...\n');

options = trainingOptions('adam', ...
    'InitialLearnRate', learning_rate, ...
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', batch_size, ...
    'Shuffle', 'every-epoch', ...
    'L2Regularization', L2_regularization, ...
    'Plots', 'training-progress', ...
    'ValidationData', {X_val, Y_val}, ...
    'ValidationFrequency', 5, ...
    'Verbose', true, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 20, ...
    'GradientThreshold', 1, ...
    'OutputNetwork', 'best-validation');

[net, info] = trainNetwork(X_train, Y_train, lgraph, options);

%% Section 5. Validation and Visualization
fprintf('Evaluating on Validation Set...\n');

% Predict
Y_pred = classify(net, X_val);

% Calculate Accuracy
acc = mean(Y_pred == Y_val);
fprintf('Validation Accuracy: %.2f%%\n', acc * 100);

% Plot Confusion Matrix
figure('Name', 'Confusion Matrix - ConvGRU', 'Color', 'w', 'Position', [200, 200, 600, 500]);
cm = confusionchart(Y_val, Y_pred);
cm.Title = sprintf('ConvGRU Confusion Matrix (Acc: %.2f%%)', acc * 100);
cm.FontName = Font_Name;
cm.FontSize = Font_Size_Basis;
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

fprintf('ConvGRU Simulation Complete.\n');