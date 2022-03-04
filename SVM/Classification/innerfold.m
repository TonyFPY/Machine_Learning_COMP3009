function [svm_best, score_best, hyperparameterInformation] = innerfold(kernel_type, X, Y)

    % constants
    NUM_OF_TRIALS = 10;

    % Min Constants
    MIN_BC = 0.1;
    MIN_POLY = 0.1;
    MIN_KS = 100;

    % Max Constants
    MAX_BC = 1;
    MAX_POLY = 2;
    MAX_KS = 400;

    SEED = 234;
    % Initialise random number generator with seed
    rng(SEED);
 
    % Vars for classification
    kernel = kernel_type;

    % To keep track and modify additional vars
    val = {};

    % To keep track of parameters
    history_boxConstraint = zeros([NUM_OF_TRIALS 1]);
    history_polyOrder = zeros([NUM_OF_TRIALS 1]);
    history_kernelScale = zeros([NUM_OF_TRIALS 1]);

    % To keep track of hyperparameter information
    precisions = zeros([NUM_OF_TRIALS 1]);
    hyperparameterInformation = cell(1,1);
    iteration = 1;

    fprintf('\n');
    disp(kernel_type);
    
    % OUTER FOLD = Number of Trials
    for i=1:NUM_OF_TRIALS
        fprintf('The %dth trial\n',i);
        % Random Search algorithm for SVM configurations    
        % Randomly generate parameters for each SVM {linear, poly, rbf}

        % Randomise/Tweak rbf Kernel Scale
        kernel_scale = ceil(MIN_KS + rand()*(MAX_KS-MIN_KS));
        history_kernelScale(i) = kernel_scale;

        % Randomise/Tweak polynomial order
        polynomial_order = MIN_POLY + rand()*(MAX_POLY-MIN_POLY);
        history_polyOrder(i) = polynomial_order;
        
        % Randomise/Tweak box constraint
        box_constraint = MIN_BC + rand()*(MAX_BC-MIN_BC);
        history_boxConstraint(i) = box_constraint;

        % Set up new SVM arguments 
        if strcmpi(kernel, 'linear_classification')
            val = {'BoxConstraint', box_constraint};
        elseif strcmpi(kernel, 'rbf_classification')
            val = {'KernelScale', kernel_scale,'BoxConstraint', box_constraint};
        elseif strcmpi(kernel, 'polynomial_classification')
            val = {'PolynomialOrder', polynomial_order,'BoxConstraint', box_constraint};
        end  

        [X_train, X_test] = get_fold(X, i, NUM_OF_TRIALS);
        [Y_train, Y_test] = get_fold(Y, i, NUM_OF_TRIALS);

        % Train SVMs
        svm = classification(X_train, Y_train, kernel, val{:});
        output = predict(svm, X_test);
       
        precision = evaluate_SVM(output, Y_test);
        precisions(i) = precision;
               
        % store hyperparameters, number of support vectors etc. into a structure
        struct.kernelType = kernel_type;
        struct.boxConstraint = box_constraint;
        struct.kernelScaleValue = kernel_scale;
        struct.polyValue = polynomial_order;
        struct.supportVectors = size(svm.SupportVectors, 1);
        struct.supportVectorsRatio = struct.supportVectors / size(X_train, 1);
        struct.precision = precision;
        hyperparameterInformation{iteration} = struct;

        iteration = iteration + 1;
        
        fprintf("Precision is %f, number of cv is %d, ratio is %f\n", precision, struct.supportVectors, struct.supportVectorsRatio);
        
        % to keep track of the best accuracy/score
        if i == 1
            score_best = precisions(i);
            svm_best = svm;
        else 
            % Check if it is a better model
            if precisions(i) > score_best
                score_best = precisions(i);
                svm_best = svm;
            end
        end    
    end

    % Plot parameters and accuracy 
    if strcmpi(kernel, 'rbf_classification')
        figure();
        scatter(history_kernelScale,precisions,36,'b','filled');
        title('Rbf - Precisions vs kernelScale');
        xlabel('kernelScale (sigma)');
        ylabel('Precision');
    elseif strcmpi(kernel, 'polynomial_classification')
        figure();
        scatter(history_polyOrder,precisions,36,'b','filled');
        title('Poly - Precisions vs poly order');
        xlabel('polyOrder (q)');
        ylabel('Precision');
    end

end
