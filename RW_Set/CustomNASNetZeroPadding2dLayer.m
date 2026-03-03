%% Custom NASNet Zero Padding 2D Layer Class Definition
% Original Author: Mathworks. 
% Improved By: JoeyBG. 
% Date: 2025-12-08. 
% Affiliate: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
%
% Introduction:
%   Custom implementation of asymmetric zero padding for improved NASNet architecture.
%   Performs 2D zero-padding on the input tensor. 
%   Unlike standard symmetric padding, 
%       this layer supports specific padding amounts for [Top, Bottom, Left, Right].
%   Essential for specific Reduction Cells in NASNet-Large.

%% Class Body
classdef CustomNASNetZeroPadding2dLayer < nnet.layer.Layer
    properties
        % Vector specifying padding: [Top, Bottom, Left, Right]
        Padding
    end

    methods
        function layer = CustomNASNetZeroPadding2dLayer(name, padding)
            % Constructor: Sets the layer name and padding configuration.
            %
            % Inputs:
            %   name    - (String/Char) Name of the layer.
            %   padding - (1x4 Array) Padding values [Top, Bottom, Left, Right].
            
            % Set layer name
            layer.Name = name;
            
            % Set layer description
            layer.Description = "Custom NASNet Zero Padding 2D: [" + ...
                join(string(padding), ' ') + "]";
            
            % Store padding parameters
            layer.Padding = padding;
        end

        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time.
            %
            % Inputs:
            %   X - Input data (Height x Width x Channels x Batch)
            %
            % Outputs:
            %   Z - Padded data
            
            % Extract padding values
            top    = layer.Padding(1);
            bottom = layer.Padding(2);
            left   = layer.Padding(3);
            right  = layer.Padding(4);
            
            % Get input dimensions
            [h, w, c, n] = size(X);
            
            % Calculate new dimensions
            newH = h + top + bottom;
            newW = w + left + right;
            
            % Initialize output tensor with zeros
            Z = zeros([newH, newW, c, n], 'like', X);
            
            % Insert original data into the padded tensor
            rowStart = top + 1;
            rowEnd   = top + h;
            colStart = left + 1;
            colEnd   = left + w;
            
            Z(rowStart:rowEnd, colStart:colEnd, :, :) = X;
        end
    end
end