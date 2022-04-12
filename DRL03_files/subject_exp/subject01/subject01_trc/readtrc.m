function [MarkerNames, MarkerPositions, pathfiletype, header_string, header_value, fieldnames, fieldsubnames, framedata] = readtrc(filename)

% to read trc files
% input  :  filename
% output :  MarkerNames   -  cell(#markers, 1)
%           MarkerPositions  - #framex by 3*#markers
%           pathfiletype    - cell such as PathFileType	4	(X/Y/Z)	C:\Users\skoo\Desktop\FK05_09MPS_FEM_MEP.trc
%           header_string   - cell such as DataRate	CameraRate	NumFrames	NumMarkers	Units	OrigDataRate	OrigDataStartFrame	OrigNumFrames
%           header_value    - cell such as 100.00	100.00	17	1	mm	100.00	1	17
%           fieldnames      - cell such as Frame#	Time	*1			
%           fieldsubnames   - cell such as 		X1	Y1	Z1	
%           framedata       - cell such as 1	0.000000	30.629880	83.564931	88.893822	
% Seungbum Koo, PhD, Chung-Ang University, Seoul, South Korea
% June 11, 2013

fid = fopen(filename);

formatstring = {'PathFileType', 'DataRate', 'Frame', '1'};

% read header information
tline = fgetl(fid);
while ischar(tline)
%     disp(tline);
    if isempty(tline)
        % read away a empty line
        tline = fgetl(fid);
        continue;
    end

    firstword = regexpi(tline, '^[\w]+', 'match');
    if isempty(firstword)
        % read away a empty line
        tline = fgetl(fid);
        continue;
    end
    
    formatidx = find(strcmpi(firstword, formatstring));

    switch(formatidx)
        case 1
            pathfiletype = regexpi(tline, '\t', 'split');
        case 2
            header_string = regexpi(tline, '\t', 'split');
            tline = fgetl(fid);
            header_value = regexpi(tline, '\t', 'split');
        case 3
            fieldnames = regexpi(tline, '\t', 'split');
            tline = fgetl(fid);
            fieldsubnames = regexpi(tline, '\t', 'split');
        case 4
            % start of frame data
            break;
    end

    tline = fgetl(fid);
end

% parse frame information
idx_NumFrames = find(strcmpi('NumFrames', header_string));
NumFrames = str2double(header_value{idx_NumFrames});

idx_NumMarkers = find(strcmpi('NumMarkers', header_string));
NumMarkers = str2double(header_value{idx_NumMarkers});

% read frame values
framenumber = 0;
framedata = cell(NumFrames, 1);

while ischar(tline)
    framenumber = framenumber + 1;
    framedata{framenumber} = regexpi(tline, '\t', 'split');
    tline = fgetl(fid);
end

fclose(fid);

% extract marker data
MarkerNames = cell(NumMarkers, 1);
for imarker = 1:NumMarkers
    MarkerNames{imarker} = fieldnames{2+(imarker-1)*3+1};
end

MarkerPositions = zeros(NumFrames, NumMarkers*3);
for iframe = 1:NumFrames
    for imarker = 1:NumMarkers
        MarkerPositions(iframe, (imarker-1)*3+1) = str2double(framedata{iframe}{2+(imarker-1)*3+1});
        MarkerPositions(iframe, (imarker-1)*3+2) = str2double(framedata{iframe}{2+(imarker-1)*3+2});
        MarkerPositions(iframe, (imarker-1)*3+3) = str2double(framedata{iframe}{2+(imarker-1)*3+3});
    end
end

