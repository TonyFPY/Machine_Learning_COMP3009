clear all;
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


linear = load('model/model_linear.mat');
rbf = load('model/model_rbf.mat');
poly = load('model/model_poly.mat');

model_linear = linear.svm_linear;
model_rbf = rbf.svm_rbf;
model_poly = poly.svm_poly;

for i = 1:kfold
    
    [X_train, X_test] = get_fold(X, i, kfold);
    [Y_train, Y_test] = get_fold(Y, i, kfold);

    % test the model, do prediction, and calculate precision for this fold
    prediction_linear = predict(model_linear, X_test);
    prediction_rbf = predict(model_rbf, X_test);
    prediction_poly = predict(model_poly, X_test);
       
    precision_linear = evaluate_SVM(prediction_linear, Y_test);
    precision_rbf = evaluate_SVM(prediction_rbf, Y_test);
    precision_poly = evaluate_SVM(prediction_poly, Y_test);
    
    Precision_linear(i) = precision_linear;
    Precision_rbf(i) = precision_rbf;
    Precision_poly(i) = precision_poly;

end

save('precision/linear_clf_precision.mat','Precision_linear');
save('precision/rbf_clf_precision.mat','Precision_rbf');
save('precision/poly_clf_precision.mat','Precision_poly');

for j = 1:kfold
    fprintf('Linear - classification rate: %f\n',Precision_linear(j));
end
for j = 1:kfold
    fprintf('RBF - classification rate: %f\n',Precision_rbf(j));
end
for j = 1:kfold
    fprintf('Poly - classification rate: %f\n',Precision_poly(j));
end