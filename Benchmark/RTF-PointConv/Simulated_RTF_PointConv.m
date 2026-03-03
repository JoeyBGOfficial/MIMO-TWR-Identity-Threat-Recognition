%% Script for Improved PointConv Network for HAR
% Original Author: Hang Xu, Yong Li, Qingran Dong, Li Liu, Jingxia Li, Jianguo Zhang, and Bingjie Wang.
% Reproduced By: JoeyBG.
% Date: 2025-12-25.
% Affiliation: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Description:
%   1 Loads RTF Point Cloud datasets from .mat files in Simulated_RTFPointSet.
%   2 Constructs the Improved PointConv architecture with 5 FEMs and Residuals.
%   3 Trains the network using Adam optimizer with built-in training plot.
%   4 Visualizes Confusion Matrix on validation set.

%% Initialization
clear all;
close all;
clc;
disp('---------- © Author: JoeyBG © ----------');

%% Parameter Definitions
% --- Path Parameters ---
data_root = 'Simulated_RTFPointSet';
% Get list of class folders
dir_info = dir(fullfile(data_root, 'P*'));
class_folders = {dir_info.name};
num_classes = length(class_folders);

% --- Training Parameters ---
train_ratio = 0.8;
batch_size = 32;
max_epochs = 80;                                                            % Sufficient epochs for convergence
learning_rate = 0.0005;                                                     % Initial learning rate
L2_regularization = 0.0001;

% --- Network Parameters ---
input_points = 1024;
input_channels = 3;                                                         % Range Time Doppler

% --- Visualization Parameters ---
Font_Name = 'Palatino Linotype'; 
Font_Size_Basis = 12;
Font_Size_Title = 14;
JoeyBG_Colormap = [0.6196 0.0039 0.2588; 0.8353 0.2431 0.3098; 0.9569 0.4275 0.2627; 0.9922 0.6824 0.3804; 0.9961 0.8784 0.5451; 1.0000 1.0000 0.7490; 0.9020 0.9608 0.5961; 0.6706 0.8667 0.6431; 0.4000 0.7608 0.6471; 0.1961 0.5333 0.7412; 0.3686 0.3098 0.6353];

%% Section 1 Data Loading and Preprocessing
fprintf('Loading Point Cloud Dataset...\n');

all_data = {};
all_labels = [];
valid_class_names = {};

% Loop through each class folder
for i = 1:num_classes
    cls_name = class_folders{i};
    cls_path = fullfile(data_root, cls_name);
    files = dir(fullfile(cls_path, '*.mat'));
    
    if isempty(files)
        continue;
    end
    
    valid_class_names{end+1} = cls_name;
    fprintf('  Loading Class %s: %d samples\n', cls_name, length(files));
    
    for k = 1:length(files)
        % Load variable Point_Cloud
        data_struct = load(fullfile(cls_path, files(k).name));
        pc_raw = data_struct.Point_Cloud; % 1024x3
        
        % Transpose to 3x1024 for Channel First processing in MATLAB
        pc_transposed = pc_raw.'; 
        
        % Min Max Normalization per sample to stabilize gradients
        pc_min = min(pc_transposed, [], 2);
        pc_max = max(pc_transposed, [], 2);
        pc_norm = (pc_transposed - pc_min) ./ (pc_max - pc_min + 1e-6);
        
        % Reshape to 3x1024x1 for Image Input Layer
        all_data{end+1} = reshape(pc_norm, [input_channels, input_points, 1]); 
        all_labels = [all_labels; categorical(string(cls_name))]; 
    end
end

% Construct 4D Array for Training
num_samples = length(all_data);
X = cat(4, all_data{:});
Y = all_labels;
classes = categories(Y);

% Random Train Validation Split
rand_idx = randperm(num_samples);
num_train = floor(train_ratio * num_samples);
idx_train = rand_idx(1:num_train);
idx_val = rand_idx(num_train+1:end);

X_train = X(:, :, :, idx_train);
Y_train = Y(idx_train);
X_val = X(:, :, :, idx_val);
Y_val = Y(idx_val);

fprintf('Dataset Prepared: Training %d, Validation %d\n', length(Y_train), length(Y_val));

%% Section 2 Improved PointConv Network Architecture
fprintf('Constructing Improved PointConv Network Graph...\n');

% Create Layer Graph
lgraph = layerGraph();

% 2.1 Input Layer
input_layer = imageInputLayer([input_channels, input_points, 1], ...
    'Name', 'Input', 'Normalization', 'none');
lgraph = addLayers(lgraph, input_layer);

% 2.2 FEM 1 MLP Dimensions 64 64
fem1 = [
    convolution2dLayer([1, 1], 64, 'Name', 'FEM1_Conv1', 'Padding', 'same')
    batchNormalizationLayer('Name', 'FEM1_BN1')
    reluLayer('Name', 'FEM1_ReLU1')
    convolution2dLayer([1, 1], 64, 'Name', 'FEM1_Conv2', 'Padding', 'same')
    batchNormalizationLayer('Name', 'FEM1_BN2')
    reluLayer('Name', 'FEM1_ReLU2')
];
lgraph = addLayers(lgraph, fem1);
lgraph = connectLayers(lgraph, 'Input', 'FEM1_Conv1');

% 2.3 FEM 2 MLP Dimensions 64 64 with Residual Connection
fem2_branch = [
    convolution2dLayer([1, 1], 64, 'Name', 'FEM2_Conv1', 'Padding', 'same')
    batchNormalizationLayer('Name', 'FEM2_BN1')
    reluLayer('Name', 'FEM2_ReLU1')
    convolution2dLayer([1, 1], 64, 'Name', 'FEM2_Conv2', 'Padding', 'same')
    batchNormalizationLayer('Name', 'FEM2_BN2')
];
lgraph = addLayers(lgraph, fem2_branch);
lgraph = addLayers(lgraph, additionLayer(2, 'Name', 'Res1_Add'));
lgraph = addLayers(lgraph, reluLayer('Name', 'Res1_ReLU'));

lgraph = connectLayers(lgraph, 'FEM1_ReLU2', 'FEM2_Conv1');         % Main path
lgraph = connectLayers(lgraph, 'FEM1_ReLU2', 'Res1_Add/in2');       % Skip connection
lgraph = connectLayers(lgraph, 'FEM2_BN2', 'Res1_Add/in1');
lgraph = connectLayers(lgraph, 'Res1_Add', 'Res1_ReLU');

% 2.4 FEM 3 MLP Dimensions 128 128
fem3 = [
    convolution2dLayer([1, 1], 128, 'Name', 'FEM3_Conv1', 'Padding', 'same')
    batchNormalizationLayer('Name', 'FEM3_BN1')
    reluLayer('Name', 'FEM3_ReLU1')
    convolution2dLayer([1, 1], 128, 'Name', 'FEM3_Conv2', 'Padding', 'same')
    batchNormalizationLayer('Name', 'FEM3_BN2')
    reluLayer('Name', 'FEM3_ReLU2')
];
lgraph = addLayers(lgraph, fem3);
lgraph = connectLayers(lgraph, 'Res1_ReLU', 'FEM3_Conv1');

% 2.5 FEM 4 MLP Dimensions 128 128 with Residual Connection
fem4_branch = [
    convolution2dLayer([1, 1], 128, 'Name', 'FEM4_Conv1', 'Padding', 'same')
    batchNormalizationLayer('Name', 'FEM4_BN1')
    reluLayer('Name', 'FEM4_ReLU1')
    convolution2dLayer([1, 1], 128, 'Name', 'FEM4_Conv2', 'Padding', 'same')
    batchNormalizationLayer('Name', 'FEM4_BN2')
];
lgraph = addLayers(lgraph, fem4_branch);
lgraph = addLayers(lgraph, additionLayer(2, 'Name', 'Res2_Add'));
lgraph = addLayers(lgraph, reluLayer('Name', 'Res2_ReLU'));

lgraph = connectLayers(lgraph, 'FEM3_ReLU2', 'FEM4_Conv1');         % Main path
lgraph = connectLayers(lgraph, 'FEM3_ReLU2', 'Res2_Add/in2');       % Skip connection
lgraph = connectLayers(lgraph, 'FEM4_BN2', 'Res2_Add/in1');
lgraph = connectLayers(lgraph, 'Res2_Add', 'Res2_ReLU');

% 2.6 FEM 5 MLP Dimensions 256 512
fem5 = [
    convolution2dLayer([1, 1], 256, 'Name', 'FEM5_Conv1', 'Padding', 'same')
    batchNormalizationLayer('Name', 'FEM5_BN1')
    reluLayer('Name', 'FEM5_ReLU1')
    convolution2dLayer([1, 1], 512, 'Name', 'FEM5_Conv2', 'Padding', 'same')
    batchNormalizationLayer('Name', 'FEM5_BN2')
    reluLayer('Name', 'FEM5_ReLU2')
];
lgraph = addLayers(lgraph, fem5);
lgraph = connectLayers(lgraph, 'Res2_ReLU', 'FEM5_Conv1');

% 2.7 Global Feature Aggregation and FC Layers
classification_head = [
    maxPooling2dLayer([1, input_points], 'Name', 'GlobalMaxPool')
    
    fullyConnectedLayer(512, 'Name', 'FC1')
    batchNormalizationLayer('Name', 'FC1_BN')
    reluLayer('Name', 'FC1_ReLU')
    dropoutLayer(0.5, 'Name', 'FC1_Drop')
    
    fullyConnectedLayer(256, 'Name', 'FC2')
    batchNormalizationLayer('Name', 'FC2_BN')
    reluLayer('Name', 'FC2_ReLU')
    dropoutLayer(0.5, 'Name', 'FC2_Drop')
    
    fullyConnectedLayer(length(classes), 'Name', 'FC_Output')
    softmaxLayer('Name', 'Softmax')
    classificationLayer('Name', 'Output_Label')
];
lgraph = addLayers(lgraph, classification_head);
lgraph = connectLayers(lgraph, 'FEM5_ReLU2', 'GlobalMaxPool');

% Analyze Constructed Network Model
analyzeNetwork(lgraph);

%% Section 3 Network Training
fprintf('Starting Training Process...\n');

% Set Plots to training-progress for built-in visualization
options = trainingOptions('adam', ...
    'InitialLearnRate', learning_rate, ...
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', batch_size, ...
    'L2Regularization', L2_regularization, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'VerboseFrequency', 10, ...
    'ValidationData', {X_val, Y_val}, ...
    'ValidationFrequency', 10, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 20, ...
    'OutputNetwork', 'best-validation');

[net, info] = trainNetwork(X_train, Y_train, lgraph, options);

%% Section 4 Confusion Matrix Visualization
fprintf('Visualizing Results...\n');

% 4.1 Confusion Matrix
Y_pred = classify(net, X_val);
overall_acc = mean(Y_pred == Y_val);
fprintf('Final Validation Accuracy: %.2f%%\n', overall_acc * 100);

fig_cm = figure('Name', 'Confusion Matrix', 'Color', 'w', 'Position', [150, 150, 700, 600]);
cm = confusionchart(Y_val, Y_pred);

cm.Title = sprintf('Confusion Matrix (Accuracy: %.2f%%)', overall_acc * 100);
cm.FontName = Font_Name;
cm.FontSize = Font_Size_Basis;
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
sortClasses(cm, classes);

fprintf('Improved PointConv Network Simulation Complete.\n');