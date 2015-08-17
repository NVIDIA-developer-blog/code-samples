function tnow = reportTime(t, messageString, scale, tbefore)

if nargin < 3
    scale = 1;
end
wait(gpuDevice);
tnow = t * scale;
disp([messageString ' = ' num2str(tnow*1e3) 'ms']);
if nargin > 3
    disp(['Speedup of ' num2str(tbefore/tnow) 'x']);
end
end