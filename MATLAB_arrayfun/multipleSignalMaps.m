% Given the setup at the start of the article, this script divides the
% antennae between 3 networks and computes a signal map for each one
% independently. While this could be done using a loop, we prefer to use
% vectorized operations to process all the data together.

% Assign network 1-3 (at random) to each antenna at every point
Network = gpuArray.randi(3, 1, M);
NetworkReplicated = repmat(Network, [N 1]);
% Sort power data and networks by power
[signalPowerSorted, I] = sort(signalPowerDecibels, 2, 'descend');
NetworkSorted = NetworkReplicated(sub2ind([N M], RowIndex(:), I(:)));
% Sort again to group by network
[~, J] = sort(reshape(NetworkSorted, [N M]), 2);
signalPowerGrouped = signalPowerSorted(J);
% Sorting original list and diff finds group boundaries as an index array
NetworkGrouped = sort(Network);
groupStartColumn = find(diff([0, NetworkGrouped]));
% Accumulated sum will allow us to find means without loops
signalPowerCum = cumsum([zeros(N,1), signalPowerGrouped], 2);
% Pick out the sum of the three most powerful antennae to get the mean
signalMap = (signalPowerCum(:,groupStartColumn+3) - ...
             signalPowerCum(:,groupStartColumn)) / 3;
drawMultipleSignalMaps(map, masts, AntennaDirection, H, signalMap)