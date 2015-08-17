function drawSignalMap(map, masts, dirn, H, signalMap)
% Draws a 2D map of received power at each gridpoint as well as a
% representation of the antennae.

hold off
N = size(map,1);
N2 = sqrt(N);
M = size(masts,1);
mapX = reshape(map(:,1), [N2 N2]);
mapY = reshape(map(:,2), [N2 N2]);
mapZ = reshape(map(:,3), [N2 N2]);
contour(mapX, mapY, mapZ); hold on;
for a = 1:M
    mastBase = masts(a,:)' - [0;0;H];
    scale = 20;
    baseline = 10;
    drawAntenna(mastBase, H*scale, dirn(:,a)*scale*baseline, scale*baseline, 5);
end
colormap hot;
caxis(gather([min(signalMap) max(signalMap)]));
surf(mapX, mapY, reshape(signalMap, size(mapX)));
shading flat;
grid off;
axis off;
end