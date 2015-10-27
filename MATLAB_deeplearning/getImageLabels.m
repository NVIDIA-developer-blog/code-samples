function imageType = getImageLabels(imset)
% Copyright (c) 2015, MathWorks, Inc.
    imageType = categorical(repelem({imset.Description}', ...
        [imset.Count], 1));
end