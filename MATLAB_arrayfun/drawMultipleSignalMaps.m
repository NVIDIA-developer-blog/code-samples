function drawMultipleSignalMaps(map, masts, AntennaDirection, H, signalMap)
% Draws signal maps stored pagewise along dimension 3 of the input
% signalMap

close;
numMaps = size(signalMap,2);
for m = 1:numMaps
subplot(1, numMaps, m);
drawSignalMap(map, masts, AntennaDirection, H, signalMap(:,m));
opos = get(gca, 'OuterPosition');
opos(1) = (m-1)/numMaps; % Left
opos(3) = 1/numMaps;     % Width
set(gca, 'OuterPosition', opos);
end
set(gcf, 'MenuBar', 'none');
oldpos = get(gcf, 'OuterPosition');
newpos = [oldpos(1:2) [oldpos(3)*1.5 oldpos(4)*0.65]*0.8];
set(gcf, 'OuterPosition', newpos);
set(gcf, 'Position', newpos);
end