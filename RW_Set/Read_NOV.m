%% Function for Readin Novasky MIMO Radar ADC Datas
% Author: Wang Tao.
% Improved By: JoeyBG.
% Date: 2025-1-15.
% Affiliate: National University of Defense Technology, Beijing Institute of Technology.
%
% Introduction:
% The code was written by Tao Wang, Ph.D. student at the National University of Defense Technology, 
%   and is used to unpack, read out, and store the raw ADC data of the Novasky production MIMO radar system as Matlab variables.
%
% References:
% None.

%% Function Body
function [adc_data_real,frame_num] = Read_NOV(fileName,packet_num,T_num,R_num,adc_num,ds_coe)

    fid = fopen(fileName, 'r+');
    
    if (fid == -1)
        uiwait(msgbox('Failed to open the file！'))
        return
    end
    
    offset = 0;
    fseek(fid,offset,'bof');
    
    adc_data = zeros(adc_num,2*packet_num/ds_coe,T_num*R_num);
    
    solving_num=packet_num*2/ds_coe;
    solved_num = 0;
    
    frame_num = [];

    for ii = 1:solving_num
        try
            frame_data = fread(fid,R_num*(adc_num+60)*T_num*ds_coe,'int16');
            frame_data_r = reshape(frame_data,R_num,(adc_num+60),T_num,ds_coe);
        
            A_permuted = permute(frame_data_r(:,1:adc_num,:,:), [1 3 2,4]);
            A_permuted = permute(A_permuted,[3,1,2,4]);

            frame_num_all = squeeze(frame_data_r(:,end,:,:));
            frame_num = [frame_num;floor(frame_num_all(1,1)/T_num)+1];
      
            adc_data_r = reshape(A_permuted, [adc_num,R_num*T_num,ds_coe]);
            adc_data(:,ii,:) = reshape(adc_data_r(:,:,1),[adc_num,1,R_num*T_num]);
            fprintf('Reading datas: %d/%d\n', ii, solving_num);
            solved_num = ii;
        catch
            disp('Passed (Reading stopped)!');
            break
        end
    end

    fclose(fid);

    adc_data_real = adc_data(:,1:solved_num,:);

end