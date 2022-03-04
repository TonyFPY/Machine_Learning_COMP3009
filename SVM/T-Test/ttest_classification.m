clear all;
clc;

load('Precision/DT_clf_precision.mat')
load('Precision/ANN_clf_precision.mat')
load('Precision/linear_clf_precision.mat');
load('Precision/rbf_clf_precision.mat');
load('Precision/poly_clf_precision.mat');


ClfNN = Precision_ANN;
ClfDT = Precision_DT;
Clflinear = Precision_linear;
Clfrbf = Precision_rbf;
Clfpoly = Precision_poly;


ClfNN = transpose(ClfNN);
ClfDT = transpose(ClfDT);
Clflinear = transpose(Clflinear);
Clfrbf = transpose(Clfrbf);
Clfpoly = transpose(Clfpoly);

[ttestRegHResults(1), ttestRegPResults(1)] = ttest2(ClfDT, ClfNN,0.05);
[ttestRegHResults(2), ttestRegPResults(2)] = ttest2(Clflinear,ClfNN,0.05);
[ttestRegHResults(3), ttestRegPResults(3)] = ttest2(Clflinear, ClfDT,0.05);
[ttestRegHResults(4), ttestRegPResults(4)] = ttest2(Clfrbf,ClfNN,0.05);
[ttestRegHResults(5), ttestRegPResults(5)] = ttest2(Clfrbf, ClfDT,0.05);
[ttestRegHResults(6), ttestRegPResults(6)] = ttest2(Clfpoly,ClfNN,0.05);
[ttestRegHResults(7), ttestRegPResults(7)] = ttest2(Clfpoly, ClfDT,0.05);
