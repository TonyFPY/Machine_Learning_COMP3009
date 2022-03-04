
kfold = 10;

% Loading dataset
disp('Loading dataset...')
data = load("dataset/PM2dot5.csv");
disp('Dataset loaded.')

% Preparing dataset
X = data(:, 2:10);
Y = data(:, 1);


[svm_linear,hyperparameterInformation_linear] = innerfold('linear_regression',X,Y);
[svm_rbf,hyperparameterInformation_rbf] = innerfold('rbf_regression',X,Y);
[svm_poly,hyperparameterInformation_poly]= innerfold('polynomial_regression',X,Y);

save('model/model_linear.mat','svm_linear');
save('model/model_rbf.mat','svm_rbf');
save('model/model_poly.mat','svm_poly');

save('history/HPInformation_linear.mat','hyperparameterInformation_linear');
save('history/HPInformation_rbf.mat','hyperparameterInformation_rbf');
save('history/HPInformation_poly.mat','hyperparameterInformation_poly');

% outerloop: 10 cross validation
for i = 1:kfold
    
    [X_train, X_test] = get_fold(X, i, kfold);
    [Y_train, Y_test] = get_fold(Y, i, kfold);


    % test the model, do prediction, and calculate root mean square (RMS) for this fold
    prediction_linear = predict(svm_linear,X_test);
    prediction_rbf = predict(svm_rbf,X_test);
    prediction_poly = predict(svm_poly,X_test);
      
    RMSE_linear_temp = sqrt(mean((prediction_linear-Y_test).^2));
    RMSE_rbf_temp = sqrt(mean((prediction_rbf-Y_test).^2));
    RMSE_poly_temp = sqrt(mean((prediction_poly-Y_test).^2));
    
    RMSE_linear(i) = RMSE_linear_temp;
    RMSE_rbf(i) = RMSE_rbf_temp;
    RMSE_poly(i) = RMSE_poly_temp;

end

average_RMSE_linear = mean(RMSE_linear);
average_RMSE_rbf = mean(RMSE_rbf);
average_RMSE_poly = mean(RMSE_poly);

supportVectors_linear = size(svm_linear.SupportVectors,1);
supportVectors_rbf = size(svm_rbf.SupportVectors,1);
supportVectors_poly = size(svm_poly.SupportVectors,1);

fprintf('Average RMSE for linear kernel is:%.2f, average support vectors numbers is: %.f\n',average_RMSE_linear,supportVectors_linear);
fprintf('Average RMSE for rbf kernel is:%.2f, average support vectors numbers is: %.f\n',average_RMSE_rbf,supportVectors_rbf);
fprintf('Average RMSE for poly kernel is:%.2f, average support vectors numbers is: %.f\n',average_RMSE_poly,supportVectors_poly);


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



