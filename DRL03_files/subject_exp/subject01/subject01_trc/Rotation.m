clear;clc;
[MarkerNames, MarkerPositions,...
    pathfiletype, header_string,...
    header_value, fieldnames,...
    fieldsubnames, framedata] = readtrc('subject01_walking17_run_cyclic.trc');

[row col] = size(MarkerPositions);

temp = zeros(row, col);
for i = 1:col/3
    temp(:,i*3-2) = MarkerPositions(:, i*3-2);
    temp(:,i*3-2+1) = MarkerPositions(:, i*3-2+2);
    temp(:,i*3-2+2) = -MarkerPositions(:, i*3-2+1);
end


%% 180 degree rotation y axis 요건 static 할 때도 항상
rotation_mat = [-1 0 0; 0 1 0; 0 0 -1]; 

temp_new = temp;

for i = 1:col/3
    for j = 1:row        
        temp_new(j,i*3-2) = -temp(j, i*3-2);
        temp_new(j,i*3-2+1) = temp(j, i*3-2+1);
        temp_new(j,i*3-2+2) = -temp(j, i*3-2+2);
    end
end
% %% +90 degree rotation y axis 실험때 -y 방향일 때 
% rotation_mat = [0 0 1; 0 1 0; -1 0 0]; 
% 
% temp_new_new = temp;
% 
% for i = 1:col/3
%     for j = 1:row        
%         temp_new_new(j,i*3-2) = temp(j, i*3-2+2);
%         temp_new_new(j,i*3-2+1) = temp(j, i*3-2+1);
%         temp_new_new(j,i*3-2+2) = -temp(j, i*3-2);
%     end
% end
%% -90 degree rotation y axis 실험때 +y 방향일 때 
rotation_mat = [0 0 -1; 0 1 0; 1 0 0]; 

temp_new_new_new = temp;

for i = 1:col/3
    for j = 1:row        
        temp_new_new_new(j,i*3-2) = -temp(j, i*3-2+2);
        temp_new_new_new(j,i*3-2+1) = temp(j, i*3-2+1);
        temp_new_new_new(j,i*3-2+2) = temp(j, i*3-2);
    end
end



