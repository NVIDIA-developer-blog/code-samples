function bboxes = findPet(frameGray, opticFlow)
% Copyright (c) 2015, MathWorks, Inc.

flow = estimateFlow(opticFlow,frameGray);
threshImage = ( flow.Magnitude > 4);
[BW_out,regions] = filterRegions(threshImage);
if(size(regions) > 0)
    bboxes = regions.BoundingBox;
else
    bboxes = [];
end
end
