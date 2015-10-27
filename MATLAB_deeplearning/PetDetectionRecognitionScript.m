% Copyright (c) 2015, MathWorks, Inc.

%% Download and and predict using a pretrained ImageNet model

% Download from MatConvNet pretrained networks repository
urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat', 'imagenet-vgg-f.mat');
cnnModel.net = load('imagenet-vgg-f.mat');

% Setup up MatConvNet, modify the path if it's installed in a different
% folder
run(fullfile('matconvnet-1.0-beta15','matlab','vl_setupnn.m'));

% Load and display an example image
imshow('dog_example.png');
img = imread('dog_example.png');

% Predict label using ImageNet trained vgg-f CNN model
label = cnnPredict(cnnModel,img);
title(label,'FontSize',20)

%% Load images from folder

% Use imageSet to manage images stored in multiple folders
imset = imageSet('pet_images','recursive');

% Preallocate arrays with fixed size for prediction
imageSize = cnnModel.net.normalization.imageSize;
trainingImages = zeros([imageSize sum([imset(:).Count])],'single');

% Load and resize images for prediction
for ii = 1:numel(imset)
    for jj = 1:imset(ii).Count
        trainingImages(:,:,:,jj) = imresize(single(read(imset(ii),jj)),imageSize(1:2));
    end
end

% Get the image labels directly from the ImageSet object
trainingLabels = getImageLabels(imset);
summary(trainingLabels)

%% Extract features using pretrained CNN

% Depending on how much memory you have on your GPU you may use a larger
% batch size. We have 400 images, I'm going to choose 200 as my batch size
cnnModel.info.opts.batchSize = 200;

% Make prediction on a CPU
[~, cnnFeatures, timeCPU] = cnnPredict(cnnModel,trainingImages,'UseGPU',false);
% Make prediction on a GPU
[~, cnnFeatures, timeGPU] = cnnPredict(cnnModel,trainingImages,'UseGPU',true);

% Compare the performance increase
bar([sum(timeCPU),sum(timeGPU)],0.5)
title(sprintf('Approximate speedup: %2.00f x ',sum(timeCPU)/sum(timeGPU)))
set(gca,'XTickLabel',{'CPU','GPU'},'FontSize',18)
ylabel('Time(sec)'), grid on, grid minor

%% Train a classifier using extracted features and calculate CV accuracy

% Train and validate a linear support vector machine (SVM) classifier.
classifierModel = fitcsvm(cnnFeatures, trainingLabels);

% 10 fold crossvalidation accuracy
cvmdl = crossval(classifierModel,'KFold',10);
fprintf('kFold CV accuracy: %2.2f\n',1-cvmdl.kfoldLoss)

%% Object Detection
% Use findPet function that was automatically generated using the 
% Image Region Analyzer App

%% Tying the workflow together
frameNumber = 0;
vr = VideoReader(fullfile('PetVideos','videoExample.mov'));
vw = VideoWriter('test.avi','Motion JPEG AVI');
opticFlow = opticalFlowFarneback;
open(vw);
while hasFrame(vr)
    % Count frames
    frameNumber = frameNumber + 1;
    
    % Step 1. Read Frame
	videoFrame = readFrame(vr);
    
    % Step 2. Detect ROI
    vFrame = imresize(videoFrame,0.25);      % Get video frame
    frameGray = rgb2gray(vFrame);            % Convert to gray for detection
    bboxes = findPet(frameGray,opticFlow);   % Find bounding boxes
    if ~isempty(bboxes)
        img = zeros([imageSize size(bboxes,1)]);
        for ii = 1:size(bboxes,1)
            img(:,:,:,ii) = imresize(imcrop(vFrame,bboxes(ii,:)),imageSize(1:2));
        end

    % Step 3. Recognize object
        % (a) Extract features using a CNN
        [~, scores] = cnnPredict(cnnModel,img,'UseGPU',true,'display',false);
        
        % (b) Predict using a trained Classifier
        label = predict(classifierModel,scores);

        % Step 4. Annotate object
        vFrame = insertObjectAnnotation(vFrame,'Rectangle',bboxes,cellstr(label),'FontSize',40);
    end
    
    % Step 5. Write video to file
      writeVideo(vw,videoFrame);

%     fprintf('Frame: %d of %d\n',frameNumber,ceil(vr.FrameRate*vr.Duration));
end
close(vw);

