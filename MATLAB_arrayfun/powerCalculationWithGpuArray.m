function signalPower = powerCalculationWithGpuArray(X, mastPos, Power, Frequency, dirn)

pathToAntenna = bsxfun(@minus, mastPos, X);
distanceSquared = sum(pathToAntenna.^2, 3);
distance = sqrt(distanceSquared);
pathLoss = (4 .* pi .* distance * Frequency ./ 3e8).^2;
signalPowerWatts = bsxfun(@rdivide, Power, pathLoss);
signalPower = 30 + 10 * log10(signalPowerWatts);
directionDotProduct = sum(bsxfun(@times, pathToAntenna, dirn), 3);
isFacing = directionDotProduct >= 0;
signalPower(~isFacing) = -inf;

end