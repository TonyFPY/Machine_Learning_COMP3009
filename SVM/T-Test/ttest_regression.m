clear all;
clc

load('RMSE/DT_reg_RMSE.mat')
load('RMSE/ANN_reg_RMSE.mat')
load('RMSE/linear_reg_RMSE.mat');
load('RMSE/rbf_reg_RMSE.mat');
load('RMSE/poly_reg_RMSE.mat');


RegNN = ANN_RMSE;
RegDT = DT_RMSE;
Reglinear = RMSE_linear;
Regrbf = RMSE_rbf;
Regpoly = RMSE_poly;

RegNN = transpose(RegNN);
RegDT = transpose(RegDT);
Reglinear = transpose(Reglinear);
Regrbf = transpose(Regrbf);
Regpoly = transpose(Regpoly);

[ttestRegHResults(1), ttestRegPResults(1)] = ttest2(RegDT, RegNN,0.05);
[ttestRegHResults(2), ttestRegPResults(2)] = ttest2(Reglinear, RegNN,0.05);
[ttestRegHResults(3), ttestRegPResults(3)] = ttest2(Reglinear, RegDT,0.05);
[ttestRegHResults(4), ttestRegPResults(4)] = ttest2(Regrbf,RegNN,0.05);
[ttestRegHResults(5), ttestRegPResults(5)] = ttest2(Regrbf, RegDT,0.05);
[ttestRegHResults(6), ttestRegPResults(6)] = ttest2(Regpoly, RegNN,0.05);
[ttestRegHResults(7), ttestRegPResults(7)] = ttest2(Regpoly, RegDT,0.05);




