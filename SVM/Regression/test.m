clear all;
clc;

% Loading test dataset
disp('Loading test dataset...')
data = load("dataset/test.csv");
disp('Test dataset loaded.')

% Preparing dataset
X = data(:, 2:10);
Y = data(:, 1);

kfold=10;

linear = load('model/model_linear.mat');
rbf = load('model/model_rbf.mat');
poly = load('model/model_poly.mat');

model_linear = linear.svm_linear;
model_rbf = rbf.svm_rbf;
model_poly = poly.svm_poly;

% outerloop: 10 cross validation
for i = 1:kfold
    
    [X_train, X_test] = get_fold(X, i, kfold);
    [Y_train, Y_test] = get_fold(Y, i, kfold);


    % test the model, do prediction, and calculate root mean square (RMS) for this fold
    prediction_linear = predict(model_linear,X_test);
    prediction_rbf = predict(model_rbf,X_test);
    prediction_poly = predict(model_poly,X_test);
      
    RMSE_linear_temp = sqrt(mean((prediction_linear-Y_test).^2));
    RMSE_rbf_temp = sqrt(mean((prediction_rbf-Y_test).^2));
    RMSE_poly_temp = sqrt(mean((prediction_poly-Y_test).^2));
    
    RMSE_linear(i) = RMSE_linear_temp;
    RMSE_rbf(i) = RMSE_rbf_temp;
    RMSE_poly(i) = RMSE_poly_temp;

end

for i=1:length(RMSE_linear)
    fprintf('The RMSE for linear is: %f\n',RMSE_linear(i));
end
fprintf("Average RMSE for linear is:%.6f\n\n",mean(RMSE_linear));

for i=1:length(RMSE_rbf)
    fprintf('The RMSE for rbf is: %f\n',RMSE_rbf(i));
end
fprintf("Average RMSE for rbf is:%.6f\n\n",mean(RMSE_rbf));

for i=1:length(RMSE_poly)
    fprintf('The RMSE for poly is: %f\n',RMSE_poly(i));
end
fprintf("Average RMSE for poly is:%.6f\n\n",mean(RMSE_poly));

save('RMSE/linear_reg_RMSE.mat','RMSE_linear');
save('RMSE/rbf_reg_RMSE.mat','RMSE_rbf');
save('RMSE/poly_reg_RMSE.mat','RMSE_poly');

