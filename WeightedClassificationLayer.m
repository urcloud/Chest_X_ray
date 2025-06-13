classdef WeightedClassificationLayer < nnet.layer.ClassificationLayer
    properties
        ClassWeights
    end

    methods
        function layer = WeightedClassificationLayer(classWeights, name)
            % 생성자
            layer.Name = name;
            layer.ClassWeights = classWeights;
            layer.Description = 'Weighted cross entropy';
        end

        function loss = forwardLoss(layer, Y, T)
            % 예측값 Y: [NumClasses x BatchSize]
            % 실제값 T: [NumClasses x BatchSize]
            % → 이 형식은 'auto' execution 환경에서 사용됨

            % 가중치를 [NumClasses x 1] 형태로 보장
            W = reshape(layer.ClassWeights, [], 1); 

            % 각 샘플의 클래스별 가중치 적용
            weightsPerSample = W' * T;  % [1 x BatchSize]

            % cross-entropy 계산
            lossPerSample = -sum(T .* log(Y + eps), 1);  % [1 x BatchSize]

            % 가중치 적용된 손실
            weightedLoss = weightsPerSample .* lossPerSample;  % [1 x BatchSize]

            % 평균 손실 (스칼라)
            loss = mean(weightedLoss, 'all');  % 반드시 스칼라 반환
        end
    end
end
