str = fileread('agent_data_mingi.json');
agent_data = jsondecode(jsondecode(str)); 

fileID1 = fopen('AMP_subject06.json','r');
mytxt1 = fscanf(fileID1,'%s');
fclose(fileID1);
mystruct1 = jsondecode(mytxt1);
% fldname1 = fieldnames(mystruct1);
% plot(mystruct1.Hip_Angle(:))

% str = fileread('AMP_subject06.json');
% expert_data = jsondecode(jsondecode(str));
expert_data = jsondecode(mystruct1);

% [1, 2, 3]  body linear vel (3)
% [4, 5, 6]  body angular vel (3)
% [7, ..., 31] joint angle (25)
% [32, ..., 56] joint velocity (25)
% [57, ..., 51] end effector (15)

% %% 01
% for i = 1:142
%     fig = figure(i);
%     subplot(2, 1, 1); 
%     plot(agent_data(1:10654, i), 'r'); hold on; plot(expert_data(1:10654, i), 'b');
%     title(sprintf('index #%d', i)), legend('agent', 'expert')
%     subplot(2, 1, 2);
%     minvalue = min([agent_data(1:10654, i); expert_data(1:10654, i)]);
%     maxvalue = max([agent_data(1:10654, i); expert_data(1:10654, i)]);
%     histf(agent_data(1:10654, i),linspace(minvalue, maxvalue, 20),'facecolor', 'none', 'alpha',.8,'edgecolor','red'), hold on;
%     histf(expert_data(1:10654, i),linspace(minvalue, maxvalue, 20),'facecolor', 'none', 'alpha',.8,'edgecolor','blue')
%     saveas(fig, sprintf('index_%03d.png', i))
% end
% %% 02
% 
% for i = 1:142
%     fig = figure(i);
%     subplot(2, 1, 1); 
%     plot(agent_data(1:5288, i), 'r'); hold on; plot(expert_data(1:5288, i), 'b');
%     title(sprintf('index #%d', i)), legend('agent', 'expert')
%     subplot(2, 1, 2);
%     minvalue = min([agent_data(1:5288, i); expert_data(1:5288, i)]);
%     maxvalue = max([agent_data(1:5288, i); expert_data(1:5288, i)]);
%     histf(agent_data(1:5288, i),linspace(minvalue, maxvalue, 20),'facecolor', 'none', 'alpha',.8,'edgecolor','red'), hold on;
%     histf(expert_data(1:5288, i),linspace(minvalue, maxvalue, 20),'facecolor', 'none', 'alpha',.8,'edgecolor','blue')
%     saveas(fig, sprintf('index_%03d.png', i))
% end
%% 03
for i = 1:142
    fig = figure(i);
    subplot(2, 1, 1); 
    plot(agent_data(1:1500, i), 'r'); hold on; plot(expert_data(1:1500, i), 'b');
    title(sprintf('index #%d', i)), legend('agent', 'expert')
    subplot(2, 1, 2);
    minvalue = min([agent_data(1:1500, i); expert_data(1:1500, i)]);
    maxvalue = max([agent_data(1:1500, i); expert_data(1:1500, i)]);
    histf(agent_data(1:1500, i),linspace(minvalue, maxvalue, 20),'facecolor', 'none', 'alpha',.8,'edgecolor','red'), hold on;
    histf(expert_data(1:1500, i),linspace(minvalue, maxvalue, 20),'facecolor', 'none', 'alpha',.8,'edgecolor','blue')
    saveas(fig, sprintf('index_%03d.png', i))
end