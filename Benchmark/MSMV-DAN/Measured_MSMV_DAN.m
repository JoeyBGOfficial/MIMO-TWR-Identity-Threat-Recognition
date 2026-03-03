%% Script for MSMV-DAN Network Reproduction for HAR
% Original Author: Yimeng Zhao, Yong Jia, Dong Huang, Li Zhang, Yao Zheng, Jianqi Wang, and Fugui Qi.
% Reproduced By: JoeyBG.
% Date: 2025-12-28.
% Affiliation: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description:
%   1. Loads 8-channel DTM datasets from 'Measured_DTMSet_Channel1' to 'Channel8'.
%   2. Preprocesses data: Resizes to 224x224 and stacks 8 views into depth dimension.
%   3. Constructs the MSMV-DAN (Dual-Layer Attention Augmented) architecture.
%   4. Implements SSA (SpectralSpace Attention) and VSA (ViewSpace Attention) logic.
%   5. Trains the network using specified parameters with visualization.
%   6. Evaluates on validation set and plots Confusion Matrix.

%% Initialization
clear all;
close all;
clc;
disp('---------- © Author: JoeyBG © ----------');

%% Parameter Definitions
% --- Path Parameters ---
root_base_name = 'Measured_DTMSet_Channel';
num_channels = 8;
% Get class names from Channel 1
dir_info = dir(fullfile([root_base_name '1'], 'P*'));
class_names = {dir_info.name};
num_classes = length(class_names);

% --- Training Parameters ---
train_ratio = 0.8;
batch_size = 32;
max_epochs = 80;
learning_rate = 0.0005;
L2_regularization = 0.0001;

% --- Network Input Parameters ---
input_height = 224;
input_width = 224;
input_depth = 8;                                                            % 8 Channels corresponding to 8 Views

% --- Visualization Parameters ---
Font_Name = 'Palatino Linotype'; 
Font_Size_Basis = 12;

%% Section 1: Multiview Data Loading and Preprocessing
fprintf('Loading and Stacking 8-Channel Multiview DTM Data...\n');

all_data = {};
all_labels = [];

% Determine total samples first to pre-allocate
for c_idx = 1:num_classes
    cls = class_names{c_idx};
    % Get file list from Channel 1 to count samples
    files = dir(fullfile([root_base_name '1'], cls, '*.mat'));
    num_samples_in_class = length(files);
    
    fprintf('  Processing Class: %s (%d samples)\n', cls, num_samples_in_class);
    
    for s_idx = 1:num_samples_in_class
        file_name = files(s_idx).name;
        
        % Temporary container for 8 views
        stacked_dtm = zeros(input_height, input_width, num_channels);
        
        % Load each channel for this specific sample
        for ch = 1:num_channels
            ch_path = fullfile([root_base_name, num2str(ch)], cls, file_name);
            if exist(ch_path, 'file')
                load(ch_path, 'Norm_DTM');
                % Resize to 224x224 as per paper requirement
                dtm_resized = imresize(Norm_DTM, [input_height, input_width]);
                stacked_dtm(:, :, ch) = dtm_resized;
            else
                error('Missing file for Channel %d: %s', ch, file_name);
            end
        end
        
        all_data{end+1} = stacked_dtm;
        all_labels = [all_labels; categorical(string(cls))];
    end
end

% Convert to 4D Array: Height x Width x Channels x Samples
num_total_samples = length(all_data);
X = cat(4, all_data{:});
Y = all_labels;
classes = categories(Y);

fprintf('Data Loaded. Shape: %d x %d x %d x %d\n', size(X));

% Train/Validation Split
rand_idx = randperm(num_total_samples);
num_train = floor(train_ratio * num_total_samples);
idx_train = rand_idx(1:num_train);
idx_val = rand_idx(num_train+1:end);

X_train = X(:, :, :, idx_train);
Y_train = Y(idx_train);
X_val = X(:, :, :, idx_val);
Y_val = Y(idx_val);

%% Section 2: Constructing MSMV-DAN Architecture
fprintf('Constructing MSMV-DAN with Dual-Layer Attention...\n');

lgraph = layerGraph();

% 2.1 Input Layer
input_layer = imageInputLayer([input_height, input_width, input_depth], ...
    'Name', 'Input', 'Normalization', 'zscore');
lgraph = addLayers(lgraph, input_layer);

% 2.2 Feature Extraction Block 1 + SSA
% Conv1
blk1 = [
    convolution2dLayer(7, 64, 'Padding', 'same', 'Stride', 2, 'Name', 'Conv1')
    batchNormalizationLayer('Name', 'BN1')
    reluLayer('Name', 'ReLU1')
    maxPooling2dLayer(3, 'Stride', 2, 'Padding', 'same', 'Name', 'Pool1')
];
lgraph = addLayers(lgraph, blk1);
lgraph = connectLayers(lgraph, 'Input', 'Conv1');

% Add SSA Module 1
lgraph = addSSABlock(lgraph, 'Pool1', 'SSA1', 64);

% 2.3 Feature Extraction Block 2 + SSA
blk2 = [
    convolution2dLayer(3, 128, 'Padding', 'same', 'Stride', 2, 'Name', 'Conv2') % Downsample
    batchNormalizationLayer('Name', 'BN2')
    reluLayer('Name', 'ReLU2')
];
lgraph = addLayers(lgraph, blk2);
lgraph = connectLayers(lgraph, 'SSA1_Out', 'Conv2');

% Add SSA Module 2
lgraph = addSSABlock(lgraph, 'ReLU2', 'SSA2', 128);

% 2.4 Feature Extraction Block 3 + SSA
blk3 = [
    convolution2dLayer(3, 256, 'Padding', 'same', 'Stride', 2, 'Name', 'Conv3') % Downsample
    batchNormalizationLayer('Name', 'BN3')
    reluLayer('Name', 'ReLU3')
];
lgraph = addLayers(lgraph, blk3);
lgraph = connectLayers(lgraph, 'SSA2_Out', 'Conv3');

% Add SSA Module 3
lgraph = addSSABlock(lgraph, 'ReLU3', 'SSA3', 256);

% 2.5 Feature Extraction Block 4
blk4 = [
    convolution2dLayer(3, 512, 'Padding', 'same', 'Name', 'Conv4')
    batchNormalizationLayer('Name', 'BN4')
    reluLayer('Name', 'ReLU4')
];
lgraph = addLayers(lgraph, blk4);
lgraph = connectLayers(lgraph, 'SSA3_Out', 'Conv4');

% 2.6 Multiview Feature Fusion & VSA
lgraph = addVSABlock(lgraph, 'ReLU4', 'VSA', 512);

% 2.7 Classification Head
clf_head = [
    globalAveragePooling2dLayer('Name', 'GAP')
    fullyConnectedLayer(256, 'Name', 'FC_Dense')
    reluLayer('Name', 'Relu_Dense')
    dropoutLayer(0.5, 'Name', 'Dropout')
    fullyConnectedLayer(num_classes, 'Name', 'FC_Out')
    softmaxLayer('Name', 'Softmax')
    classificationLayer('Name', 'Output')
];
lgraph = addLayers(lgraph, clf_head);
lgraph = connectLayers(lgraph, 'VSA_Out', 'GAP');

% Analyze
analyzeNetwork(lgraph);

%% Section 3: Network Training
fprintf('Starting Training Process...\n');

options = trainingOptions('adam', ...
    'InitialLearnRate', learning_rate, ...
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', batch_size, ...
    'L2Regularization', L2_regularization, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'ValidationData', {X_val, Y_val}, ...
    'ValidationFrequency', 10, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 20);

[net, info] = trainNetwork(X_train, Y_train, lgraph, options);

%% Section 4: Validation and Confusion Matrix
fprintf('Visualizing Results...\n');

Y_pred = classify(net, X_val);
accuracy = mean(Y_pred == Y_val);
fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);

figure('Name', 'Confusion Matrix - MSMV-DAN', 'Color', 'w', 'Position', [100, 100, 700, 600]);
cm = confusionchart(Y_val, Y_pred);
cm.Title = ['MSMV-DAN Confusion Matrix (Acc: ' num2str(accuracy*100, '%.2f') '%)'];
cm.FontName = Font_Name;
cm.FontSize = Font_Size_Basis;
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

fprintf('Reproduction Complete.\n');

%% Helper Functions
function lg = addSSABlock(lg, input_layer_name, block_name, num_channels)
    % SpectralSpace Attention (SSA) = Channel Attention + Spatial Attention
    
    % 1. Channel Attention Branch
    gap_name = [block_name '_GAP'];
    fc1_name = [block_name '_CA_FC1'];
    relu_name = [block_name '_CA_ReLU'];
    fc2_name = [block_name '_CA_FC2'];
    sig_name = [block_name '_CA_Sig'];
    scale_name = [block_name '_CA_Scale'];
    
    ca_layers = [
        globalAveragePooling2dLayer('Name', gap_name)
        fullyConnectedLayer(floor(num_channels/4), 'Name', fc1_name) % Reduction ratio
        reluLayer('Name', relu_name)
        fullyConnectedLayer(num_channels, 'Name', fc2_name)
        sigmoidLayer('Name', sig_name)
    ];
    
    lg = addLayers(lg, ca_layers);
    lg = connectLayers(lg, input_layer_name, gap_name);
    
    % Multiplication for CA
    lg = addLayers(lg, multiplicationLayer(2, 'Name', scale_name));
    lg = connectLayers(lg, input_layer_name, [scale_name '/in1']);
    lg = connectLayers(lg, sig_name, [scale_name '/in2']);
    
    % 2. Spatial Attention Branch    
    conv_sa_name = [block_name '_SA_Conv'];
    sig_sa_name = [block_name '_SA_Sig'];
    mult_sa_name = [block_name '_Out']; % Final Output of SSA block
    
    % Compressing channels to 1 to find "Where" is important
    sa_layers = [
        convolution2dLayer(7, 1, 'Padding', 'same', 'Name', conv_sa_name)
        sigmoidLayer('Name', sig_sa_name)
    ];
    
    lg = addLayers(lg, sa_layers);
    lg = connectLayers(lg, scale_name, conv_sa_name);
    
    % Apply Spatial Attention
    lg = addLayers(lg, multiplicationLayer(2, 'Name', mult_sa_name));
    lg = connectLayers(lg, scale_name, [mult_sa_name '/in1']);
    lg = connectLayers(lg, sig_sa_name, [mult_sa_name '/in2']);
end

function lg = addVSABlock(lg, input_layer_name, block_name, num_channels)
    % ViewSpace Attention (VSA)    
    gap_name = [block_name '_GAP'];
    fc1_name = [block_name '_FC1'];
    relu_name = [block_name '_ReLU'];
    fc2_name = [block_name '_FC2'];
    sig_name = [block_name '_Sig'];
    out_name = [block_name '_Out'];
    
    vsa_layers = [
        globalAveragePooling2dLayer('Name', gap_name)
        fullyConnectedLayer(floor(num_channels/8), 'Name', fc1_name)
        reluLayer('Name', relu_name)
        fullyConnectedLayer(num_channels, 'Name', fc2_name)
        sigmoidLayer('Name', sig_name)
    ];
    
    lg = addLayers(lg, vsa_layers);
    lg = connectLayers(lg, input_layer_name, gap_name);
    
    % Scale features by View Weights
    lg = addLayers(lg, multiplicationLayer(2, 'Name', out_name));
    lg = connectLayers(lg, input_layer_name, [out_name '/in1']);
    lg = connectLayers(lg, sig_name, [out_name '/in2']);
end