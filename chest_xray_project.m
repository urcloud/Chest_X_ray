clear; clc; close all;

%% 1. 데이터 경로 설정 (상대 경로로 설정)
datasetPath = fullfile(pwd, 'chest_xray', 'train');

%% 2. 데이터 로드
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

%% 3. 라벨 분포 확인
tbl = countEachLabel(imds);
disp(tbl);

%% 4. 이미지 예시 확인
figure;
perm = randperm(numel(imds.Files), 10);
for i = 1:10
    subplot(2,5,i);
    img = readimage(imds, perm(i));
    imshow(img);
    title(string(imds.Labels(perm(i))));
end

%% 5. 이미지 전처리 (크기 통일)
imds.ReadFcn = @(filename)imresize(imread(filename), [224 224]);

% 크기 확인
img = readimage(imds, 1);
disp(size(img));
