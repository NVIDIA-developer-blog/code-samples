function [classLabel, scores, batchTime] = cnnPredict(cnnModel,predImage,varargin)
% Copyright (c) 2015, MathWorks, Inc.

% Parse inputs
p = inputParser;
addParameter(p,'outputLayer',numel(cnnModel.net.layers),@isnumeric);
addParameter(p,'UseGPU',false,@islogical);
addParameter(p,'display',true,@islogical);
parse(p,varargin{:});

% Get batch size and number of images
if ~isfield(cnnModel,'info')
    cnnModel.info.opts.batchSize = 1;
end
batchSize = cnnModel.info.opts.batchSize;
n_obs = size(predImage,4);
isTapLayer = p.Results.outputLayer < numel(cnnModel.net.layers);

if isTapLayer
    cnnModel.net.layers(p.Results.outputLayer+1:end) = [];
else
    cnnModel.net.layers{end} = struct('type', 'softmax');
end

% Preallocate scores
resTemp = vl_simplenn(cnnModel.net, cnnPreprocess(predImage(:,:,:,1)), [], []);
scores = zeros([size(resTemp(end).x), n_obs]);

% Move model to GPU if requested
if p.Results.UseGPU
    cnnModel.net = vl_simplenn_move(cnnModel.net,'gpu');
end

% Make predictions
batchNumber = 0;
numBatches = ceil(n_obs/batchSize);
batchTime = zeros(numBatches,1);
if p.Results.display
    disp(' ')
    fprintf('Using GPU: %s\n',mat2str(p.Results.UseGPU))
    fprintf('Number of images: %d\n',n_obs)
    fprintf('Number of batches: %d\n',numBatches)
    fprintf('Number of layers in the Network: %d\n',numel(cnnModel.net.layers))
    disp('-------------------------------------')
end
for ii = 1:batchSize:n_obs
    tic
    idx = ii:min(ii+batchSize-1,n_obs);
	batchImages = predImage(:,:,:,idx);
    im = cnnPreprocess(batchImages);
    
	% Move batch to GPU if requested
    if p.Results.UseGPU
        im = gpuArray(im);
    end
    train_res = vl_simplenn(cnnModel.net, im, [], []);
    scores(:,:,:,idx) = squeeze(gather(train_res(end).x));
    batchNumber = batchNumber + 1;
    batchTime(batchNumber) = toc;
    if p.Results.display
        fprintf('Batch: %2d/%d. Execution time: %2.4f\n',batchNumber,numBatches,batchTime(batchNumber))
    end
end

if p.Results.display
	fprintf('Avg. execution time/batch:   %2.4f\n',mean(batchTime))
    disp('-------------------------------------')
    fprintf('Total execution time:        %2.4f\n',sum(batchTime))
    disp('-------------------------------------')
end

if isTapLayer
    classLabel = [];
else
    scores = squeeze(gather(scores))';
    [~, labelId] = max(scores,[],2);
%     classLabel = categorical(cnnModel.net.classes.description(labelId)');
    classLabel = cnnModel.net.classes.description(labelId)';
end

function im = cnnPreprocess(batchImages)
    % Preprocess images
    im = single(batchImages);
    im = imresize(im, cnnModel.net.normalization.imageSize(1:2));
	im = bsxfun(@minus,im,cnnModel.net.normalization.averageImage);
end

end
