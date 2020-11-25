function [trainingInput, trainingLabels, validationInput, validationLabels] = crossValidation(x, y, folds, fold)
% Cross validation
%
% Input:
%	  x: features
%	  y: labels
%	  folds: number of folds
%	  fold: sequence number of the fold
%
% Output:
%	[trainingInput, trainingLabels, validationInput, validationLabels]

    featureSize = size(x);
    foldSize = floor(featureSize(1) / folds);

    shift = foldSize * (fold - 1);
    shiftedX = circshift(x, shift);
    shiftedY = circshift(y, shift);

    validationInput = shiftedX(1:foldSize, :);
    validationLabels = shiftedY(1:foldSize, :);

    trainingInput = shiftedX(foldSize + 1:end, :);
    trainingLabels = shiftedY(foldSize + 1:end, :);
end
