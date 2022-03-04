% Clear command window and variables
clear;
clc;

kfold = 10;

% Loading dataset
disp('Loading dataset...')
data = load("iris_new.csv");
disp('Dataset loaded.')

% Preparing dataset
randidx = randperm(size(data, 1));
X = data(randidx, 1:4);
Y = data(randidx, 5);

[svm_linear,best,hyperparameterInformation_linear] = innerfold('linear_classification', X, Y);
fprintf("The best precision is %f\n", best);
[svm_rbf,best,hyperparameterInformation_rbf] = innerfold('rbf_classification', X, Y);
fprintf("The best precision is %f\n", best);
[svm_poly,best,hyperparameterInformation_poly] = innerfold('polynomial_classification', X, Y);
fprintf("The best precision is %f\n", best);

save('model/model_linear.mat', 'svm_linear');
save('model/model_rbf.mat', 'svm_rbf');
save('model/model_poly.mat', 'svm_poly');

% outerloop: 10 cross validation
for i = 1:kfold
    
    [X_train, X_test] = get_fold(X, i, kfold);
    [Y_train, Y_test] = get_fold(Y, i, kfold);


    % test the model, do prediction, and calculate precision for this fold
    prediction_linear = predict(svm_linear, X_test);
    prediction_rbf = predict(svm_rbf, X_test);
    prediction_poly = predict(svm_poly, X_test);
    
   
    precision_linear = evaluate_SVM(prediction_linear, Y_test);
    precision_rbf = evaluate_SVM(prediction_rbf, Y_test);
    precision_poly = evaluate_SVM(prediction_poly, Y_test);
    
    Precision_linear(i) = precision_linear;
    Precision_rbf(i) = precision_rbf;
    Precision_poly(i) = precision_poly;

end

average_Precision_linear = mean(Precision_linear);
average_Precision_rbf = mean(Precision_rbf);
average_Precision_poly = mean(Precision_poly);

supportVectors_linear = size(svm_linear.SupportVectors,1);
supportVectors_rbf = size(svm_rbf.SupportVectors,1);
supportVectors_poly = size(svm_poly.SupportVectors,1);

fprintf('\n\nAverage Precision for linear kernel is:%.2f, average support vectors numbers is: %.f\n',average_Precision_linear,supportVectors_linear);
fprintf('Average Precision for rbf kernel is:%.2f, average support vectors numbers is: %.f\n',average_Precision_rbf,supportVectors_rbf);
fprintf('Average Precision for poly kernel is:%.2f, average support vectors numbers is: %.f\n',average_Precision_poly,supportVectors_poly);


printSVandRation(hyperparameterInformation_linear);
printSVandRation(hyperparameterInformation_rbf);
printSVandRation(hyperparameterInformation_poly);


function printSVandRation(hyperparameterInformation)
    for i=1:length(hyperparameterInformation)
        SV = hyperparameterInformation{i}.supportVectors;
        ratio = hyperparameterInformation{i}.supportVectorsRatio;
        kernel_type = hyperparameterInformation{i}.kernelType;

        fprintf('%s %d,the number of support vectors:%d, and of %.2f%% of the training data available\n',kernel_type,i,SV,ratio*100);
    end
    fprintf('\n');
end

