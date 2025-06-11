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
    imageInputLayer([224 224 1], 'Name', 'input')

    convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv_1')
    batchNormalizationLayer('Name', 'batchnorm_1')
    reluLayer('Name', 'relu_1')

    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_1')

    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv_2')
    batchNormalizationLayer('Name', 'batchnorm_2')
    reluLayer('Name', 'relu_2')

    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_2')
    dropoutLayer(0.25, 'Name', 'dropout_1')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv_3')
    batchNormalizationLayer('Name', 'batchnorm_3')
    reluLayer('Name', 'relu_3')
    
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_3')
    dropoutLayer(0.5, 'Name', 'dropout')

    fullyConnectedLayer(numel(unique(imdsTrain.Labels)), 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')
];


%% 6. 학습 옵션 설정
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 20, ...
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


%% 9. 특징 추출용 중간층 선택
featureLayer = 'relu_3';
features = activations(net, imdsTest, featureLayer, 'OutputAs', 'rows');
labels = imdsTest.Labels;
minLength = min(size(features, 1), numel(labels));
features = features(1:minLength, :);
labels = labels(1:minLength);


%% 10. PCA
[score, ~, ~, ~, explained] = pca(features);
minLength = min(size(score, 1), numel(labels));
score = score(1:minLength, :);
labels = labels(1:minLength);
figure;
gscatter(score(:,1), score(:,2), labels);
title(sprintf('PCA 결과 (설명된 분산: %.2f%% + %.2f%%)', explained(1), explained(2)));
xlabel('PC1'); ylabel('PC2'); grid on;


%% 11. t-SNE 시각화
rng(1); % 고정 시드
Y = tsne(features);
minLength = min(size(Y, 1), numel(labels));
Y = Y(1:minLength, :);
labels = labels(1:minLength);
figure;
gscatter(Y(:,1), Y(:,2), labels);
title('t-SNE 시각화');
xlabel('Dimension 1');
ylabel('Dimension 2');
grid on;