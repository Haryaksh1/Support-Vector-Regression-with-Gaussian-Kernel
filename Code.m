% Load the data
% Assuming the data is in a CSV file named 'supervised_modeling_sample.csv'
data = readtable('Supervised modeling data - Sheet1.csv');

% Extract features (X) and target variable (y)
X = table2array(data(:, 1:end-1));
y = table2array(data(:, end));

% Normalize/standardize the features (important for SVR)
[X_norm, mu, sigma] = zscore(X);

% Split the data into training and testing sets (80% train, 20% test)
rng(42); % For reproducibility
cv = cvpartition(size(X_norm, 1), 'HoldOut', 0.2);
X_train = X_norm(cv.training, :);
y_train = y(cv.training);
X_test = X_norm(cv.test, :);
y_test = y(cv.test);

% Parameter ranges for grid search
C_values = [0.1, 1, 10, 100, 1000]; % Regularization parameter
sigma_values = [0.01, 0.1, 1, 10, 100]; % Kernel parameter (sigma)

% Initialize variables to store best parameters and performance
best_C = 0;
best_sigma = 0;
best_mse = Inf;
results = zeros(length(C_values), length(sigma_values));

% Cross-validation for parameter selection
k = 5; % k-fold cross-validation
cv_partition = cvpartition(length(y_train), 'KFold', k);

% Progress tracking
total_iterations = length(C_values) * length(sigma_values);
iteration = 0;

fprintf('Starting parameter grid search...\n');

% Grid search for best parameters
for i = 1:length(C_values)
    C = C_values(i);
    
    for j = 1:length(sigma_values)
        sigma = sigma_values(j);
        
        iteration = iteration + 1;
        fprintf('Testing parameters %d/%d: C = %f, sigma = %f\n', ...
                iteration, total_iterations, C, sigma);
        
        % Initialize array to store MSE for each fold
        mse_folds = zeros(k, 1);
        
        % Perform k-fold cross-validation
        for fold = 1:k
            % Get training and validation indices for this fold
            train_idx = cv_partition.training(fold);
            val_idx = cv_partition.test(fold);
            
            % Split the data
            X_cv_train = X_train(train_idx, :);
            y_cv_train = y_train(train_idx);
            X_cv_val = X_train(val_idx, :);
            y_cv_val = y_train(val_idx);
            
            % Train SVR model
            svr_model = fitrsvm(X_cv_train, y_cv_train, ...
                               'KernelFunction', 'gaussian', ...
                               'BoxConstraint', C, ...
                               'KernelScale', sigma, ...
                               'Standardize', false); % Already standardized
            
            % Make predictions
            y_pred = predict(svr_model, X_cv_val);
            
            % Calculate MSE for this fold
            mse_folds(fold) = mean((y_cv_val - y_pred).^2);
        end
        
        % Average MSE across all folds
        avg_mse = mean(mse_folds);
        results(i, j) = avg_mse;
        
        % Check if this is the best combination of parameters
        if avg_mse < best_mse
            best_mse = avg_mse;
            best_C = C;
            best_sigma = sigma;
        end
    end
end

fprintf('Best parameters found: C = %f, sigma = %f with MSE = %f\n', ...
        best_C, best_sigma, best_mse);

% Visualize the grid search results
figure;
[C_grid, sigma_grid] = meshgrid(C_values, sigma_values);
surf(log10(C_grid), log10(sigma_grid), results');
xlabel('log10(C)');
ylabel('log10(sigma)');
zlabel('Mean Squared Error');
title('Parameter Selection for SVR with Gaussian Kernel');
colorbar;

% Train final SVR model with the best parameters
final_model = fitrsvm(X_train, y_train, ...
                     'KernelFunction', 'gaussian', ...
                     'BoxConstraint', best_C, ...
                     'KernelScale', best_sigma, ...
                     'Standardize', false); % Already standardized

% Make predictions on the test set
y_pred_test = predict(final_model, X_test);

% Evaluate the final model
mse_test = mean((y_test - y_pred_test).^2);
rmse_test = sqrt(mse_test);
mae_test = mean(abs(y_test - y_pred_test));
r2_test = 1 - sum((y_test - y_pred_test).^2) / sum((y_test - mean(y_test)).^2);

fprintf('\nTest set performance metrics:\n');
fprintf('MSE: %f\n', mse_test);
fprintf('RMSE: %f\n', rmse_test);
fprintf('MAE: %f\n', mae_test);
fprintf('R^2: %f\n', r2_test);

% Visualize the predictions
figure;
scatter(y_test, y_pred_test);
hold on;
min_val = min([y_test; y_pred_test]);
max_val = max([y_test; y_pred_test]);
plot([min_val, max_val], [min_val, max_val], 'r--');
xlabel('Actual Values');
ylabel('Predicted Values');
title('SVR with Gaussian Kernel: Actual vs. Predicted');
grid on;

