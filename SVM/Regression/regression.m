function model = regression(X, Y, classification_type, varargin)
    % X Features
    % Y Targets
    %
    % CLASSIFICATION_TYPE Kernal function to run, valid inputs are:
    %   linear_classification, linear_regression
    %   polynomial_classification, polynomial_regression
    %   rbg_classification, rbf_regression
    %
    % ADDITIONAL_PARAMS Kernal hyperparameters, valid inputs are:
    %   KernalScale, PolynomialOrder, Epsilon
    %
    % Valid pairings of CLASSIFICATION_TYPE and ADDITIONAL_PARAMS are:
    %   linear_classification => NONE
    %   polynomial_classification => PolynomialOrder
    %   rbf_classification => KernalScale
    %   *_regression => **PARAM + Epsilon
    %   * any CLASSIFICATION_TYPE
    %   ** Valid pairings of CLASSIFICATION_TYPE => ADDITIONAL_PARAM
    
    p = inputParser;
    
    % Default values for SVM
    defaultKernalScale = 1;
    defaultPolynomialOrder = 3;
    defaultEpsilon = iqr(Y) / 13.49;
    defaultBoxConstraint = 1;
    
    % Set up inputParser 'p'
    addParameter(p, 'KernelScale', defaultKernalScale, @isnumeric);
    addParameter(p, 'PolynomialOrder', defaultPolynomialOrder, @isnumeric);
    addParameter(p, 'Epsilon', defaultEpsilon, @isnumeric);
    addParameter(p, 'BoxConstraint', defaultBoxConstraint, @isnumeric);
    
    % Override values in inputParser from argument
    parse(p, varargin{:});
    
    if strcmpi(classification_type, 'linear_regression')
        model = fitrsvm(X, Y, 'KernelFunction', 'linear', 'BoxConstraint', p.Results.BoxConstraint, 'Epsilon', p.Results.Epsilon);         
    elseif strcmpi(classification_type, 'rbf_regression')
        model = fitrsvm(X, Y, 'KernelFunction', 'rbf', 'KernelScale', p.Results.KernelScale, 'BoxConstraint', p.Results.BoxConstraint, 'Epsilon', p.Results.Epsilon);
    elseif strcmpi(classification_type, 'polynomial_regression')
        model = fitrsvm(X, Y, 'KernelFunction', 'polynomial', 'PolynomialOrder', p.Results.PolynomialOrder, 'BoxConstraint', p.Results.BoxConstraint, 'Epsilon', p.Results.Epsilon); 
        
    end
end