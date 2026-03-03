%% Through-the-Wall Radar Human Activity Recognition Script: Multi-Stream Deep Ensemble Processor
% Original Author: JoeyBG. 
% Improved By: JoeyBG. 
% Date: 2025-12-08. 
% Affiliate: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Introduction:
%   This script implements a multi-modal deep learning framework for through-the-wall radar (TWR)-based
%       human activity recognition (HAR) integrating time-range, doppler-time, 
%       range-doppler, and riemannian feature streams.
%   Methodology:
%       Synchronized datastores are split using stratified sampling to ensure
%           alignment across four distinct modalities.
%       Independent xception networks are trained via transfer learning for
%           each data stream.
%       A soft voting ensemble strategy fuses probability vectors to derive
%           the final classification result.
%   Output:
%       Trained neural network models and evaluation metrics including
%           confusion matrices and accuracy scores.
%
% Files to process:
%   Datasets located in RW_RTM_Set, RW_DTM_Set, RW_RDM_Set, and RW_Feature_Set.

%% Initialization of MATLAB Script
clear all;
close all;
clc;
disp('---------- © Author: JoeyBG © ----------');
disp('   Multi-Stream Radar Activity Recognition');
disp('=============================================================');

% Configuration of data paths and parameters
Data_Paths = ["RW_RTM_Set", "RW_DTM_Set", "RW_RDM_Set", "RW_Feature_Set"];
Modality_Names = ["RTM", "DTM", "RDM", "Feature"];
Num_Modalities = length(Data_Paths);

% Training split ratio
Train_Ratio = 0.8;

% Define input size requirements for Xception network
Input_Size = [299, 299, 3]; 

% Set training batch size
Batch_Size = 32; 

% Training hyperparameters
Max_Epochs = 80;
Learn_Rate = 0.0005;                                                        % Initial learning rate for transfer learning
Execution_Env = 'auto';                                                     % Automatic gpu detection
Validation_Freq = 10;                                                       % Frequency of validation checks

% Visualization parameters
Vis_Params.Font_Name = 'Palatino Linotype';                                 % Font name used for plotting
Vis_Params.Font_Size_Basis = 15;                                            % Base font size
Vis_Params.Font_Size_Axis = 16;                                             % Font size for axis labels
Vis_Params.Font_Size_Title = 18;                                            % Font size for the title
Vis_Params.Font_Weight_Basis = 'normal';                                    % Base font weight
Vis_Params.Font_Weight_Axis = 'normal';                                     % Font weight for axis labels
Vis_Params.Font_Weight_Title = 'bold';                                      % Font weight for the title

%% Synchronized Data Loading and Splitting
% Principle: Load all datastores but generate split indices based on the first one
fprintf('Step 1: Loading and Synchronizing Datasets...\n');
IMDS_All = cell(1, Num_Modalities);

% Load raw datastores for each modality
for i = 1:Num_Modalities
    if ~exist(Data_Paths(i), 'dir')
        error('Folder not found: %s', Data_Paths(i));
    end
    IMDS_All{i} = imageDatastore(Data_Paths(i), ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');
    
    % Verify file count alignment across streams
    if i > 1
        if numel(IMDS_All{i}.Files) ~= numel(IMDS_All{1}.Files)
            error('Mismatch in file counts between %s and %s.', Modality_Names(1), Modality_Names(i));
        end
    end
end

% Generate stratified split indices based on the first modality labels
labels = IMDS_All{1}.Labels;
cv_part = cvpartition(labels, 'HoldOut', 1 - Train_Ratio);
train_indices = cv_part.training;
val_indices   = cv_part.test;

% Apply split indices to all datastores
Data_Train = cell(1, Num_Modalities);
Data_Val   = cell(1, Num_Modalities);

for i = 1:Num_Modalities
    Data_Train{i} = subset(IMDS_All{i}, train_indices);
    Data_Val{i}   = subset(IMDS_All{i}, val_indices);
    fprintf('  > Modality [%s]: Train=%d, Val=%d files prepared.\n', ...
        Modality_Names(i), numel(Data_Train{i}.Files), numel(Data_Val{i}.Files));
end

% Specific category extraction
Classes = categories(labels);
Num_Classes = numel(Classes);

%% Individual Model Training
Nets = cell(1, Num_Modalities);
Val_Scores = cell(1, Num_Modalities);                                       % Storage for validation probabilities

for i = 1:Num_Modalities
    fprintf('\n-------------------------------------------------------------\n');
    fprintf('Step 2.%d: Training Xception for Modality: %s\n', i, Modality_Names(i));
    fprintf('-------------------------------------------------------------\n');
    
    % Prepare augmented datastores with resizing
    aug_Train = augmentedImageDatastore(Input_Size, Data_Train{i}, ...
        'ColorPreprocessing', 'gray2rgb');
    aug_Val   = augmentedImageDatastore(Input_Size, Data_Val{i}, ...
        'ColorPreprocessing', 'gray2rgb');
    
    % Retrieve modified xception architecture
    lgraph = get_xception_architecture(Num_Classes);
    
    % Configure training options
    opts = trainingOptions('adam', ...
        'MiniBatchSize', Batch_Size, ...
        'MaxEpochs', Max_Epochs, ...
        'InitialLearnRate', Learn_Rate, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', aug_Val, ...
        'ValidationFrequency', Validation_Freq, ...
        'Verbose', false, ...
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', Execution_Env,...
        'OutputNetwork', 'best-validation');
    
    % Execute network training
    tic;
    [trainedNet, info] = trainNetwork(aug_Train, lgraph, opts);
    trainTime = toc;
    
    Nets{i} = trainedNet;
    fprintf('  > Training Finished in %.1fs. Final Validation Acc: %.2f%%\n', ...
        trainTime, info.ValidationAccuracy(end));
    
    % Compute probabilities for ensemble integration
    fprintf('  > Calculating Probabilities for Ensemble...\n');
    probs = predict(trainedNet, aug_Val);
    Val_Scores{i} = probs;
end

%% Ensemble Learning via Soft Voting Strategy
fprintf('\n=============================================================\n');
fprintf('Step 3: Performing Deep Ensemble Soft Voting\n');
fprintf('=============================================================\n');

% Average the probability vectors across all models
Summed_Scores = zeros(size(Val_Scores{1}));
for i = 1:Num_Modalities
    Summed_Scores = Summed_Scores + Val_Scores{i};
end
Average_Scores = Summed_Scores / Num_Modalities;

% Determine final class based on maximum probability
[~, max_idx] = max(Average_Scores, [], 2);
Pred_Ensemble = Classes(max_idx);
Pred_Ensemble = categorical(Pred_Ensemble, Classes);
True_Labels = Data_Val{1}.Labels; 

%% Visualization and Metrics Evaluation
fprintf('\nStep 4: Evaluating Ensemble Results...\n');

% Calculate overall accuracy
acc = mean(Pred_Ensemble == True_Labels);
fprintf('  > Single Model Accuracies (approx):\n');
for i = 1:Num_Modalities
    [~, idx_i] = max(Val_Scores{i}, [], 2);
    pred_i = categorical(Classes(idx_i), Classes);
    acc_i = mean(pred_i == True_Labels);
    fprintf('    - %s: %.2f%%\n', Modality_Names(i), acc_i * 100);
end
fprintf('  > --------------------------------------\n');
fprintf('  > ENSEMBLE FINAL ACCURACY: %.2f%%\n', acc * 100);
fprintf('  > --------------------------------------\n');

% Generate confusion matrix with consistent visualization parameters
hFig = figure('Name', 'Multi-Modal Xception Ensemble Results', 'Color', 'w', 'Position', [100, 100, 800, 600]);
cm = confusionchart(True_Labels, Pred_Ensemble);
cm.Title = sprintf('Ensemble - Acc: %.2f%%', acc * 100);
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

% Apply font styling from visualization parameters
cm.FontName = Vis_Params.Font_Name;
cm.FontSize = Vis_Params.Font_Size_Basis;

% Define output folder
Save_Dir = pwd;

% Create timestamped filename
Timestamp = string(datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss'));
FileName = "ConfusionMatrix_" + Timestamp + ".png";
FullPath = fullfile(Save_Dir, FileName);

% Export figure with high resolution
try
    exportgraphics(hFig, FullPath, 'Resolution', 600);
    fprintf('  > Confusion Matrix Saved: %s\n', FullPath);
catch ME
    warning('Failed to save image using exportgraphics. Trying saveas...');
    saveas(hFig, fullfile(Save_Dir, "ConfusionMatrix_" + Timestamp + ".fig"));
end

disp('Processing Completed.');

%% Core Function: Architecture Modification
function lgraph = get_xception_architecture(num_classes)
    % get_xception_architecture - Prepare xception network for custom classification
    try
        net = xception;
    catch
        error('Xception support package not installed. Please install "Deep Learning Toolbox Model for Xception Network" from Add-Ons.');
    end
    lgraph = layerGraph(net);
    
    % Define new fully connected layer
    newFCLayer = fullyConnectedLayer(num_classes, ...
        'Name', 'new_predictions', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10);
        
    % Define new classification output layer
    newClassLayer = classificationLayer('Name', 'new_classoutput');
    
    % Replace existing layers with custom layers
    lgraph = replaceLayer(lgraph, 'predictions', newFCLayer);
    lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newClassLayer);
    
    disp('  > Xception architecture modified for custom classes.');
end