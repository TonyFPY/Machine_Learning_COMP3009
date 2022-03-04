function [svm_best,hyperparameterInformation] = innerfold(kernel_type,X,Y)

    % constants
    NUM_OF_TRIALS = 10;

    % Min Constants
    MIN_BC = 0.1;
    MIN_EPSILON = 0.01;
    MIN_POLY = 0.1;
    MIN_KS = 100;

    % Max Constants
    MAX_BC = 1;
    MAX_EPSILON = 0.1;
    MAX_POLY = 2;
    MAX_KS = 400;

    SEED = 220;
    % Initialise random number generator with seed
    rng(SEED);


    % Variables to interface with SVM.m

    % Vars for regression
    kernel = kernel_type;

    % To keep track and modify additional vars
    val = {};

    % To keep track of parameters
    history_boxConstraint = zeros([NUM_OF_TRIALS 1]);
    history_epsilon = zeros([NUM_OF_TRIALS 1]);
    history_polyOrder = zeros([NUM_OF_TRIALS 1]);
    history_kernelScale = zeros([NUM_OF_TRIALS 1]);

    % To keep track of average Errors
    average_err = zeros([NUM_OF_TRIALS 1]);
    
    hyperparameterInformation = cell(1,1);
    iteration = 1;

    disp(kernel_type);
    % OUTER FOLD = Number of Trials
    for i=1:NUM_OF_TRIALS
        fprintf('The %dth trial\n',i);
        % Random Search algorithm for SVM configurations    
        % Randomly generate parameters for each SVM {linear, poly, rbf}

        % Randomise/Tweak Epsilon
        epsilon = MIN_EPSILON + rand()*(MAX_EPSILON-MIN_EPSILON);
        history_epsilon(i) = epsilon;

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
        if strcmpi(kernel, 'linear_regression')
            val = {'BoxConstraint', box_constraint,'Epsilon', epsilon};
        elseif strcmpi(kernel, 'rbf_regression')
            val = {'BoxConstraint', box_constraint,'Epsilon', epsilon, 'KernelScale', kernel_scale};
        elseif strcmpi(kernel, 'polynomial_regression')
            val = {'BoxConstraint', box_constraint,'Epsilon', epsilon, 'PolynomialOrder', polynomial_order};
        end  
%           % Set up new SVM arguments 
%         if strcmpi(kernel, 'linear_regression')
%             val = {'BoxConstraint', box_constraint,'Epsilon', 0.0579};
%         elseif strcmpi(kernel, 'rbf_regression')
%             val = {'BoxConstraint', box_constraint,'Epsilon', 0.0424, 'KernelScale', 213};
%         elseif strcmpi(kernel, 'polynomial_regression')
%             val = {'BoxConstraint', box_constraint,'Epsilon', 0.0513, 'PolynomialOrder', 0.7077};
%         end 

        [X_train, X_test] = get_fold(X, i, NUM_OF_TRIALS);
        [Y_train, Y_test] = get_fold(Y, i, NUM_OF_TRIALS);

        % Train SVMs
        svm = regression(X_train, Y_train, kernel, val{:});
        output = predict(svm, X_test);

        RMSE = sqrt(mean((output-Y_test).^2));

        % Determine the performance of this randomly generated model
        % Calculate the average mean squared error
        average_err(i) = RMSE;
        
        if strcmpi(kernel, 'linear_regression')
            sv_linear(i) = size(svm.SupportVectors,1);
        elseif strcmpi(kernel, 'rbf_regression')
            sv_rbf(i) = size(svm.SupportVectors,1);
        elseif strcmpi(kernel, 'polynomial_regression')
            sv_poly(i) = size(svm.SupportVectors,1);
        end
        
        % store hyperparameters, number of support vectors, and the average RMS into a structure
        struct.kernelType = kernel_type;
        struct.boxConstraint = box_constraint;
        struct.epsilonValue = epsilon;
        struct.kernelScaleValue = kernel_scale;
        struct.polyValue = polynomial_order;
        struct.supportVectors = size(svm.SupportVectors,1);
        struct.supportVectorsRatio = struct.supportVectors / size(X_train,1);
        struct.rmse = RMSE;
        hyperparameterInformation{iteration} = struct;

        iteration = iteration + 1;
        
        
        % to keep track of the best accuracy/score
        if i == 1
            score_best = average_err(i);
            svm_best = svm;
        else 
            % Check if it is a better model
            if average_err(i) < score_best
                score_best = average_err(i);
                svm_best = svm;
            end
        end    
    end

    % Plot parameters and accuracy

    figure();
    scatter(history_epsilon, average_err,36,'b','filled');
    title('RMSE vs Epsilon');
    xlabel('Epsilon');
    ylabel('Error rate');

    if strcmpi(kernel, 'rbf_regression')
        figure();
        scatter(history_kernelScale, average_err,36,'b','filled');
        title('Rbf - RMSE vs kernelScale');
        xlabel('kernelScale');
        ylabel('RMSE');
    elseif strcmpi(kernel, 'polynomial_regression')
        figure();
        scatter(history_polyOrder, average_err,36,'b','filled');
        title('Poly - RMSE vs poly order');
        xlabel('polyOrder');
        ylabel('RMSE');
    end

end
