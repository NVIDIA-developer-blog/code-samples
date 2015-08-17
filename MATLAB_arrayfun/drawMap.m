function drawMap(map, masts, dirn, H)
% Draws a 3D map of the topological data and antenna positions

close all
N = size(map,1);
N2 = sqrt(N);
M = size(masts,1);
mapX = reshape(map(:,1), [N2 N2]);
mapY = reshape(map(:,2), [N2 N2]);
mapZ = reshape(map(:,3), [N2 N2]);
mesh(gather(mapX), gather(mapY), gather(mapZ));
view(-41,41);
hold on;
for a = 1:M
    mastBase = masts(a,:)' - [0;0;H];
    scale = 20;
    baseline = 10;
    drawAntenna(mastBase, H*scale, dirn(:,a)*scale*baseline, scale*baseline, 5);
end
axis off
set(gca, 'Position', [0 0 1 1], 'ActivePositionProperty', 'position');
end
