% % fileID1 = fopen('AMP_subject03_exp.json','r');
% % fileID2 = fopen('agent_data_subject03_run.json','r');
% fileID1 = fopen('AMP_subject06_01.json','r');
% fileID2 = fopen('agent_data_run.json','r');
% 
% mytxt1 = fscanf(fileID1,'%s');
% fclose(fileID1);
% mystruct1 = jsondecode(mytxt1);
% fldname1 = fieldnames(mystruct1);
% 
% mytxt2 = fscanf(fileID2,'%s');
% fclose(fileID2);
% mystruct2 = jsondecode(mytxt2);
% fldname2 = fieldnames(mystruct2);
% 
% 
% [row1, col1] = size(mystruct1.expert);
% [row2, col2] = size(mystruct2.agent);
% 
% 
% data = [];
% 
% for i = 1:row1
%     data = [data; mystruct1.expert(i,:)];
% end
% 
% for i = 1:2000
%     data = [data; mystruct2.agent(i,:)];
% end
% 
% 
% myjson.expert = data;
% 
% txt = jsonencode(myjson);
% 
% fileID = fopen('AMP_subject06_aug_v2.json','w');
% fwrite(fileID, txt, 'char');
% fclose(fileID);

%%

fileID1 = fopen('AMP_subject06_aug_v2.json','r');

mytxt1 = fscanf(fileID1,'%s');
fclose(fileID1);
mystruct1 = jsondecode(mytxt1);
fldname1 = fieldnames(mystruct1);

[row1, col1] = size(mystruct1.expert);







