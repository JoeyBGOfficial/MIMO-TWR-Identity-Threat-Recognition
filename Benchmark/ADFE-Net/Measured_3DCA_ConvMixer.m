%% Script for 3DCA-ConvMixer Network for TWR HAR
% Original Author: Longzhen Tang, Shisheng Guo, Jiachen Li, Junda Zhu, Guolong Cui, Lingjiang Kong, and Xiaobo Yang.
% Reproduced By: JoeyBG.
% Date: 2025-12-29.
% Affiliation: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description:
%   1. Loads "Enhanced_DTM" datasets from 'Measured_Enhanced_DTMSet'.
%   2. Preprocesses data: Resizing to 224x224 and Normalization.
%   3. Constructs the 3DCA-ConvMixer architecture:
%      - Patch Embedding.
%      - Stacked Blocks with DCU, 3D Attention, Residuals, and PCU.
%   4. Trains the network and visualizes the Confusion Matrix.

%% Initialization
clear all;
close all;
clc;
disp('---------- © Author: JoeyBG © ----------');

%% Parameter Definitions
% --- Path Parameters ---
data_root = 'Measured_Enhanced_DTMSet';
dir_info = dir(fullfile(data_root, 'P*'));
class_folders = {dir_info.name};
num_classes = length(class_folders);

% --- Data Dimensions ---
input_size = [224, 224, 1];                                                 % Resize spectrograms to fixed size for Patch Embedding
num_channels = 1;

% --- Architecture Parameters ---
embedding_dim = 256;                                                        % Dimension 'h'
patch_size = 7;                                                             % Kernel 'p'
depth = 8;                                                                  % Number of blocks 'Q'
kernel_size = 9;                                                            % Conv kernel size 'k'

% --- Training Parameters ---
train_ratio = 0.8;
batch_size = 32;
max_epochs = 80;                                                            % As per paper Table II settings
learning_rate = 0.0005;                                                     % Initial LR
learn_rate_drop_factor = 0.7;
learn_rate_drop_period = 50;

% --- Visualization Parameters ---
Font_Name = 'Palatino Linotype'; 
Font_Size_Basis = 12;

%% Section 1. Data Loading and Preprocessing
fprintf('Loading Enhanced DTM Dataset...\n');

all_data = {};
all_labels = [];

for i = 1:num_classes
    cls_name = class_folders{i};
    cls_path = fullfile(data_root, cls_name);
    files = dir(fullfile(cls_path, '*.mat'));
    
    if isempty(files)
        continue;
    end
    
    fprintf('  Loading Class %s: %d samples\n', cls_name, length(files));
    
    for k = 1:length(files)
        % Load variable 'Enhanced_DTM' (Freq x Time matrix)
        data_struct = load(fullfile(cls_path, files(k).name));
        dtm_raw = data_struct.Enhanced_DTM;
        
        % Resize to [224, 224] using Bicubic interpolation for network input
        dtm_resized = imresize(dtm_raw, input_size(1:2));
        
        % Normalize to [0, 1]
        dtm_norm = (dtm_resized - min(dtm_resized(:))) / (max(dtm_resized(:)) - min(dtm_resized(:)) + 1e-6);
        
        % Store as 4D array compatible cell [H, W, C]
        all_data{end+1} = reshape(dtm_norm, input_size); 
        all_labels = [all_labels; categorical(string(cls_name))]; 
    end
end

% Convert to Matrix
num_samples = length(all_data);
X = cat(4, all_data{:});
Y = all_labels;
classes = categories(Y);

% Split Data
rand_idx = randperm(num_samples);
num_train = floor(train_ratio * num_samples);
idx_train = rand_idx(1:num_train);
idx_val = rand_idx(num_train+1:end);

X_train = X(:, :, :, idx_train);
Y_train = Y(idx_train);
X_val = X(:, :, :, idx_val);
Y_val = Y(idx_val);

fprintf('Dataset Prepared: Train %d, Val %d\n', length(Y_train), length(Y_val));

%% Section 2. 3DCA-ConvMixer Architecture Construction
fprintf('Constructing 3DCA-ConvMixer Network...\n');

lgraph = layerGraph();

% 2.1 Input & Patch Embedding
input_layer = imageInputLayer(input_size, 'Name', 'Input', 'Normalization', 'none');
lgraph = addLayers(lgraph, input_layer);

patch_embed = [
    convolution2dLayer(patch_size, embedding_dim, ...
        'Stride', patch_size, 'Name', 'PatchEmbed_Conv')
    geluLayer('Name', 'PatchEmbed_GELU')
    batchNormalizationLayer('Name', 'PatchEmbed_BN')
];
lgraph = addLayers(lgraph, patch_embed);
lgraph = connectLayers(lgraph, 'Input', 'PatchEmbed_Conv');

last_layer_name = 'PatchEmbed_BN';

% 2.2 Stacked 3DCA-ConvMixer Blocks
% Structure: Residual -> [DCU -> 3DCA] + Input -> PCU
for d = 1:depth
    blk_name = sprintf('Blk%02d_', d);
    
    % Depthwise Conv Unit (DCU)
    dcu_layers = [
        groupedConvolution2dLayer(kernel_size, 1, embedding_dim, ...
            'Padding', 'same', ...
            'Name', [blk_name 'DCU_Conv'])
        geluLayer('Name', [blk_name 'DCU_GELU'])
        batchNormalizationLayer('Name', [blk_name 'DCU_BN'])
    ];
    lgraph = addLayers(lgraph, dcu_layers);
    lgraph = connectLayers(lgraph, last_layer_name, [blk_name 'DCU_Conv']);
    
    % 3D Coordinate Attention (3DCA) Approximation   
    % Branch for Attention
    avg_pool_name = [blk_name 'Attn_GlobalPool'];
    fc1_name      = [blk_name 'Attn_FC1'];
    relu_name     = [blk_name 'Attn_ReLU'];
    fc2_name      = [blk_name 'Attn_FC2'];
    sigmoid_name  = [blk_name 'Attn_Sigmoid'];
    scale_name    = [blk_name 'Attn_Scale'];
    
    attn_layers = [
        globalAveragePooling2dLayer('Name', avg_pool_name)
        fullyConnectedLayer(round(embedding_dim/4), 'Name', fc1_name)
        reluLayer('Name', relu_name)
        fullyConnectedLayer(embedding_dim, 'Name', fc2_name)
        sigmoidLayer('Name', sigmoid_name)
    ];
    lgraph = addLayers(lgraph, attn_layers);
    
    % Connect DCU output to Attention
    lgraph = connectLayers(lgraph, [blk_name 'DCU_BN'], avg_pool_name);
    
    % Multiplication Layer
    mult_layer = multiplicationLayer(2, 'Name', scale_name);
    lgraph = addLayers(lgraph, mult_layer);
    lgraph = connectLayers(lgraph, [blk_name 'DCU_BN'], [scale_name '/in1']);
    lgraph = connectLayers(lgraph, sigmoid_name, [scale_name '/in2']);
    
    % Residual Connection
    add_layer_name = [blk_name 'Res_Add'];
    lgraph = addLayers(lgraph, additionLayer(2, 'Name', add_layer_name));
    
    % Connect Scaled Output to Add
    lgraph = connectLayers(lgraph, scale_name, [add_layer_name '/in1']);
    % Connect Block Input (Residual) to Add
    lgraph = connectLayers(lgraph, last_layer_name, [add_layer_name '/in2']);
    
    % Pointwise Conv Unit (PCU)
    pcu_layers = [
        convolution2dLayer(1, embedding_dim, 'Name', [blk_name 'PCU_Conv'])
        geluLayer('Name', [blk_name 'PCU_GELU'])
        batchNormalizationLayer('Name', [blk_name 'PCU_BN'])
    ];
    lgraph = addLayers(lgraph, pcu_layers);
    lgraph = connectLayers(lgraph, add_layer_name, [blk_name 'PCU_Conv']);
    
    last_layer_name = [blk_name 'PCU_BN'];
end

% 2.3 Classification Head
head_layers = [
    globalAveragePooling2dLayer('Name', 'GlobalPool')
    fullyConnectedLayer(num_classes, 'Name', 'FC_Out')
    softmaxLayer('Name', 'Softmax')
    classificationLayer('Name', 'Output_Label')
];
lgraph = addLayers(lgraph, head_layers);
lgraph = connectLayers(lgraph, last_layer_name, 'GlobalPool');

% Analyze
analyzeNetwork(lgraph);

%% Section 3. Network Training
fprintf('Starting Training...\n');

options = trainingOptions('adam', ...
    'InitialLearnRate', learning_rate, ...
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', batch_size, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'VerboseFrequency', 10, ...
    'ValidationData', {X_val, Y_val}, ...
    'ValidationFrequency', 10, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', learn_rate_drop_factor, ...
    'LearnRateDropPeriod', learn_rate_drop_period, ...
    'OutputNetwork', 'best-validation');

[net, info] = trainNetwork(X_train, Y_train, lgraph, options);

%% Section 4. Results Visualization
fprintf('Visualizing Results...\n');

% 4.1 Prediction
Y_pred = classify(net, X_val);
acc = mean(Y_pred == Y_val);
fprintf('Validation Accuracy: %.2f%%\n', acc * 100);

% 4.2 Confusion Matrix
figure('Name', 'ADFE-Net Confusion Matrix', 'Color', 'w', 'Position', [200, 200, 700, 600]);
cm = confusionchart(Y_val, Y_pred);
cm.Title = sprintf('3DCA-ConvMixer (Acc: %.2f%%)', acc * 100);
cm.FontName = Font_Name;
cm.FontSize = Font_Size_Basis;
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
sortClasses(cm, classes);

fprintf('Training and Evaluation Complete.\n');