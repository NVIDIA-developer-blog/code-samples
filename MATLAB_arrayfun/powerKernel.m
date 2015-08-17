function signalPower = powerKernel(X, Y, Z, mastX, mastY, mastZ, Power, Frequency, dirnX, dirnY, dirnZ)
% Arrayfun kernel for computing received power at [X;Y;Z] from antenna at
% [mastX; mastY; mastZ]

pathX = mastX - X; pathY = mastY - Y; pathZ = mastZ - Z;
distanceSquared = pathX.*pathX + pathY.*pathY + pathZ.*pathZ;
distance = sqrt(distanceSquared);
pathLoss = (4 .* pi .* distance .* Frequency ./ 3e8).^2;
signalPowerWatts = Power ./ pathLoss;
signalPower = 30 + 10 * log10(signalPowerWatts);
directionDotProduct = pathX*dirnX + pathY*dirnY + pathZ*dirnZ;
if directionDotProduct < 0
    signalPower = -inf;
end

end