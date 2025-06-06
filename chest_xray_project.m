clear; clc; close all;

%% 1. 데이터 경로 설정
trainPath = fullfile(pwd, 'chest_xray', 'train');
valPath = fullfile(pwd, 'chest_xray', 'val');
testPath = fullfile(pwd, 'chest_xray', 'test');

%% 2. 데이터 로드
imdsTrain = imageDatastore(trainPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

imdsValidation = imageDatastore(valPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

imdsTest = imageDatastore(testPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

%% 3. 전처리 함수 (이미지 크기 통일)
resizeFcn = @(filename) imresize(im2gray(imread(filename)), [224 224]);
imdsTrain.ReadFcn = resizeFcn;
imdsValidation.ReadFcn = resizeFcn;
imdsTest.ReadFcn = resizeFcn;

%% 4. 이미지 크기 확인
img = readimage(imdsTrain, 1);
disp("이미지 크기:");
disp(size(img));

%% 5. CNN 아키텍처 정의
layers = [
    imageInputLayer([224 224 1])

    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(numel(unique(imdsTrain.Labels)))
    softmaxLayer
    classificationLayer
];

%% 6. 학습 옵션 설정
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 5, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%% 7. 학습
net = trainNetwork(imdsTrain, layers, options);

%% 8. 테스트 데이터로 평가
YPred = classify(net, imdsTest);
YTest = imdsTest.Labels;

accuracy = sum(YPred == YTest) / numel(YTest);
fprintf('테스트 정확도: %.2f%%\n', accuracy * 100);
