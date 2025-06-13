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


%% 클래스 가중치 계산
labelCount = countEachLabel(imdsTrain);
numNormal = labelCount{strcmp(labelCount.Label, 'NORMAL'), 'Count'};
numPneumonia = labelCount{strcmp(labelCount.Label, 'PNEUMONIA'), 'Count'};

weightNormal = 1;
weightPneumonia = numNormal / numPneumonia;
classWeights = [weightNormal, weightPneumonia];


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
    WeightedClassificationLayer(classWeights, 'weighted_output')
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


%% 12. 정밀도(Precision), 재현율(Recall), F1-score 계산 및 출력

[confMat, order] = confusionmat(YTest, YPred);

numClasses = size(confMat, 1);
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1score = zeros(numClasses, 1);

for i = 1:numClasses
    TP = confMat(i, i);
    FP = sum(confMat(:, i)) - TP;
    FN = sum(confMat(i, :)) - TP;
    
    precision(i) = TP / (TP + FP + eps);
    recall(i) = TP / (TP + FN + eps);
    f1score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i) + eps);
end

disp('=== 분류 성능 (클래스별) ===');
for i = 1:numClasses
    fprintf('클래스: %s\n', string(order(i)));
    fprintf('  Precision: %.2f%%\n', precision(i) * 100);
    fprintf('  Recall   : %.2f%%\n', recall(i) * 100);
    fprintf('  F1-Score : %.2f%%\n\n', f1score(i) * 100);
end

macroPrecision = mean(precision);
macroRecall = mean(recall);
macroF1 = mean(f1score);
fprintf('=== 전체 평균 (Macro-Averaged) ===\n');
fprintf('  Precision: %.2f%%\n', macroPrecision * 100);
fprintf('  Recall   : %.2f%%\n', macroRecall * 100);
fprintf('  F1-Score : %.2f%%\n', macroF1 * 100);
