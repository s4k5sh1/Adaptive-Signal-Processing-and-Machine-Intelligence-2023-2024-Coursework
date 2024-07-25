%% 4.1

clear; close all;

% Load the dataset
load('time-series.mat');

% Plot original time series with mean
figure;
subplot(2,1,1);
plot(y, 'LineWidth', 1); hold on;
plot([1 length(y)], [mean(y) mean(y)], 'r--', 'LineWidth', 1);
title('Original Time Series with Mean');
xlabel('Sample'); ylabel('Magnitude');
legend('y[n]','E\{y[n]\}','Interpreter','latex')
grid on;

% Remove the mean from the time series
yAdjusted = y - mean(y);

% Re-plot adjusted time series
subplot(2,1,2);
plot(yAdjusted, 'LineWidth', 1); hold on;
yline(mean(yAdjusted), 'r--','LineWidth', 1);
title('Adjusted Time Series');
xlabel('Sample'); ylabel('Magnitude');
legend('y[n]-E\{y[n]\}','E\{y[n]\}','Interpreter','latex')
grid on;

% LMS Prediction
mu = 1e-5; % Learning rate
order = 4; % AR model order
[N, pred, err, W] = performLMS(yAdjusted, mu, order);

% Plot prediction results
figure;
subplot(2,1,1);
plot(yAdjusted); hold on; plot(pred, 'r');
title('One Step Ahead LMS Prediction vs. Adjusted Series');
legend('y[n]-E\{y[n]\}', 'AR(4) Prediction');
xlabel('Sample'); ylabel('Magnitude');

subplot(2,1,2);
plot(800:length(yAdjusted), yAdjusted(800:end)); hold on;
plot(800:length(pred), pred(800:end), 'r');
title('LMS Prediction vs. Adjusted Series (Zoomed In)');
xlabel('Sample'); ylabel('Magnitude');
legend('y[n]-E\{y[n]\}', 'AR(4) Prediction');

% Error analysis
MSE = pow2db(mean(abs(err).^2)); %In dB
predictionGain = 10 * log10(var(pred) / var(err));

%% 4.2


dynamical_perceptron_output = tanh(pred);

% Now plot the zero-mean signal against the output of the dynamical perceptron
figure;
subplot(2,1,1);
plot(yAdjusted, 'LineWidth', 1); hold on; % Original zero-mean signal
plot(dynamical_perceptron_output, 'r-', 'LineWidth', 1); % Perceptron output
title('Zero-Mean Signal vs. Dynamical Perceptron Output');
xlabel('Sample');
ylabel('Magnitude');
legend('Zero-Mean Signal', 'Dynamical Perceptron Output');
grid on;

subplot(2,1,2);
plot(800:length(yAdjusted), yAdjusted(800:end)); hold on;
plot(800:length(dynamical_perceptron_output), dynamical_perceptron_output(800:end), 'r');
title('Zero-Mean Signal vs. Dynamical Perceptron Output (Zoomed In)');
xlabel('Sample'); ylabel('Magnitude');
legend('y[n]-E\{y[n]\}', 'Dynamical Perceptron Output');

%% 4.3

close all; 
clear all; 
clc;

%% Loading data
load 'time-series.mat';

% Removing the mean of the time-series
y_zero_mean = y - mean(y);

%% Calculating MSE and Prediction Gain for different 'a'
a = (1:0.05:100);
MSE = zeros(1, length(a));
Rp = zeros(1, length(a));

mu = 10^(-7);
num_coef = 4; % Assuming AR(4) process

for i = 1:length(a)
    % Implementing LMS for prediction with modified function
    [~, error, ~] = ScaledTanh(y_zero_mean, mu, num_coef, a(i));
    
    % Mean Square Error (dB)
    MSE(i) = pow2db(mean(abs(error).^2)); % In dB
    
    % Prediction Gain (dB)
    Rp(i) = pow2db(var(y_zero_mean) / var(error));
end

% Finding optimal values of 'a'
[MSE_min, index_MSE] = min(MSE);
[Rp_max, index_Rp] = max(Rp);

% Using Optimal 'a' for Prediction
a_opt = mean([a(index_MSE), a(index_Rp)]);

% Implementing prediction with optimal 'a'
[y_hat_opt, error_opt, ~] = ScaledTanh(y_zero_mean, mu, num_coef, a_opt);

% Calculating MSE and Rp for optimal 'a' in db
MSE_opt = pow2db(mean(abs(error_opt).^2));
Rp_opt = pow2db(var(y_hat_opt) / var(error_opt));

%% Plotting Prediction with Optimal 'a'
figure;
subplot(2,1,1);
plot(y_zero_mean, 'LineWidth', 1);
hold on;
plot(y_hat_opt, 'r', 'LineWidth', 1);
title('Zero-Mean Signal vs. Optimal Prediction with Scaled tanh Activation', 'FontSize', 11);
xlabel('Sample', 'FontSize', 11);
ylabel('Magnitude', 'FontSize', 11);
legend('y[n]-E\{y[n]\}', 'Scaled tanh Prediction', 'Interpreter', 'latex');
grid on;

subplot(2,1,2);
plot(800:1000, y_zero_mean(800:1000), 'LineWidth', 1);
hold on;
plot(800:1000, y_hat_opt(800:1000), 'r', 'LineWidth', 1);
title('Zero-Mean Signal vs. Optimal Prediction with Scaled tanh Activation (Zoomed in)', 'FontSize', 11);
xlabel('Sample', 'FontSize', 11);
ylabel('Magnitude', 'FontSize', 11);
legend('y[n]-E\{y[n]\}', 'Scaled tanh Prediction', 'Interpreter', 'latex');
grid on;


%% 4.4


close all; clear all; clc;

%% Loading data
load 'time-series.mat';

%% Calculating MSE and Prediction Gain for different a
a = 1:0.01:100;
MSE = zeros(1, length(a));
Rp = zeros(1, length(a));

mu = 10^(-7);
num_coef = 4;

%% Considering bias in prediction
MSE_bias = zeros(1, length(a));
Rp_bias = zeros(1, length(a));

for i = 1:length(a)
    % Implementing LMS for prediction without bias
    [y_hat, error, ~] = tanh_scale_bias(y, mu, num_coef, a(i));
    
    % Mean Square Error (dB)
    MSE(i) = pow2db(mean(abs(error).^2));
    
    % Prediction Gain (dB)
    Rp(i) = pow2db(var(y_hat) / var(error));
    
    % Implementing LMS for prediction with bias
    [y_hat_bias, error_bias, ~] = tanh_scale_bias(y, mu, num_coef, a(i));
    
    % Mean Square Error with bias (dB)
    MSE_bias(i) = pow2db(mean(abs(error_bias).^2));
    
    % Prediction Gain with bias (dB)
    Rp_bias(i) = pow2db(var(y_hat_bias) / var(error_bias));
end

% Finding optimal 'a' with and without bias
[MSE_min, index_MSE] = min(MSE);
[Rp_max, index_Rp] = max(Rp);
a_opt = mean([a(index_MSE), a(index_Rp)]);

[MSE_min_bias, index_MSE_bias] = min(MSE_bias);
[Rp_max_bias, index_Rp_bias] = max(Rp_bias);
a_opt_bias = mean([a(index_MSE_bias), a(index_Rp_bias)]);

% Implementing LMS for prediction with optimal 'a' and bias
[y_hat_opt, error_opt, ~] = tanh_scale_bias(y, mu, num_coef, a_opt_bias);

% Plotting
figure;
subplot(2,1,1)
plot(y,'Linewidth',1)
hold on
plot(y_hat_opt,'r', 'Linewidth',1)
grid on

title('Original Signal vs. Optimal Prediction with Biased and Scaled tanh Activation','Fontsize',11)
xlabel('Sample', 'FontSize', 11);
ylabel('Magnitude', 'FontSize', 11);
legend('y[n]','Biased and Scaled tanh LMS estimate','Interpreter','latex')


subplot(2,1,2)
plot(y,'Linewidth',1)
hold on
plot(y_hat_opt,'r','Linewidth',1)
grid on
title('Original Signal vs. Optimal Prediction with Biased and Scaled tanh Activation (Zoomed In)','Fontsize',11)
xlabel('Sample', 'FontSize', 11);
ylabel('Magnitude', 'FontSize', 11);
legend('y[n]','Biased and Scaled tanh LMS estimate','Interpreter','latex')
xlim([800 1000]);




%% 4.5


close all; 
clear all; 
clc;

% Loading data
load 'time-series.mat';

% Considering bias in prediction
a = (1:0.01:100);
mu = 10^(-7);
num_coef = 4; % Assuming AR(4) process

% Initialize arrays for MSE and Prediction Gain
MSE_pre = zeros(1, length(a));
Rp_pre = zeros(1, length(a));
w_init_mtx = zeros(length(a), num_coef + 1);

% Pre-training of coefficients
N_samples = 20;
N_epochs = 100;
y_pre = y(1:N_samples); % Vector used for pre-training

% LMS pre-training loop
for i = 1:length(a)
    w_init = zeros(num_coef + 1, 1); % Reset initial weights for each 'a'
    for epoch_ind = 1:N_epochs
        [~, ~, wpre] = pretraining(y_pre, mu, num_coef, a(i), w_init); 
        w_init = wpre(:, end);
    end
    w_init_mtx(i, :) = w_init; % Store initial weights
    
    [y_hat_pre, error_pre, ~] = pretraining(y, mu, num_coef, a(i), w_init);
    
    MSE_pre(i) = pow2db(mean(abs(error_pre).^2));
    Rp_pre(i) = pow2db(var(y_hat_pre) / var(error_pre));
end

% Finding optimal scale
[MSE_min_pre, index_MSE_pre] = min(MSE_pre);
[Rp_max_pre, index_Rp_pre] = max(Rp_pre);
a_opt_pre = mean([a(index_MSE_pre), a(index_Rp_pre)]);

% Find corresponding initial weights
w_init_opt = w_init_mtx(round(0.5 * (index_MSE_pre + index_Rp_pre)), :);

% Implementing LMS for prediction using optimal scale
[y_hat_opt, ~, ~] = pretraining(y, mu, num_coef, a_opt_pre, w_init_opt);

% Plotting results
figure;
subplot(2,1,1)
plot(y, 'Linewidth', 1)
hold on
plot(y_hat_opt, 'r', 'Linewidth', 1)
grid on
title('One-step ahead Prediction, Produced with Bias and Pre-Trained Weights','Fontsize',11)
xlabel('Sample','Fontsize',11)
ylabel('Magnitude','Fontsize',11)
legend('y[n]', 'LMS with pretraining', 'Interpreter', 'latex')


subplot(2,1,2)
plot(y, 'Linewidth', 1)
hold on
plot(y_hat_opt, 'r', 'Linewidth', 1)
grid on
title('One-step ahead Prediction, Produced with Bias and Pre-Trained Weights (Zoomed In)','Fontsize',11)
xlabel('Sample','Fontsize',11)
ylabel('Magnitude','Fontsize',11)
legend('y[n]', 'LMS with pretraining', 'Interpreter', 'latex')
xlim([800 1000])

%%
function [N, pred, err, W] = performLMS(signal, mu, order)
    N = length(signal);
    pred = zeros(1, N);
    err = zeros(1, N);
    W = zeros(order, N+1); % Weight matrix
    
    for n = order+1:N
        x = flip(signal(n-order:n-1)); % Ensuring 'x' is a column vector
        pred(n) = W(:, n)' * x; % Attempt to perform dot product
        err(n) = signal(n) - pred(n);
        W(:, n+1) = W(:, n) + mu * err(n) * x;
    end
    W = W(:, 2:end); % Adjust weight matrix to remove initial column
end




function [predicted, predictionError, coefficients] = ScaledTanh(signal, learningRate, order, scalingFactor)
    % Ensure signal is a column vector
    signal = signal(:);
    signalLength = length(signal); 
    inputData = zeros(order, signalLength);
    
    % Preparing inputData matrix with shifted values of the signal
    for i = 1:order
        inputData(i, i+1:end) = signal(1:end-i);
    end
    
    % Initialize error, coefficients, and prediction arrays
    predictionError = zeros(signalLength, 1);
    coefficients = zeros(order, signalLength); % Adjusted to match updates
    predicted = zeros(signalLength, 1);
    
    % Iterate over the signal to update coefficients and make predictions
    for j = order+1:signalLength
        % Forming the current input vector from inputData
        currentInput = inputData(:, j);
        
        % Weighted input for current prediction using AR model coefficients
        weightedInput = coefficients(:, j-order)' * currentInput; % Adjusted indexing for coefficients
        
        % Apply scaling factor and tanh activation to weighted input
        predicted(j) = scalingFactor * tanh(weightedInput);
        
        % Compute prediction error
        predictionError(j) = signal(j) - predicted(j);
        
        % Derivative of tanh function for weight update
        derivative = scalingFactor * (1 - tanh(weightedInput)^2);
        
        % Update AR model coefficients based on the prediction error
        coefficients(:, j-order+1) = coefficients(:, j-order) + learningRate * predictionError(j) * derivative * currentInput;
    end
    
    % Adjust coefficients matrix to discard the unused initial columns
    coefficients = coefficients(:, 1:signalLength - order);
end





function [x_hat, error, weights] = tanh_scale_bias(input_x, mu, filter_order, scale)
    N = length(input_x);
    x_n = zeros(filter_order, N);
    
    % Obtaining x_n
    for k = 1:filter_order
        x_n(k, :) = [zeros(1, k), input_x(1:N - k)'];
    end
    
    % Constructing augmented input - to account for bias
    aug_x_n = [ones(1, size(x_n, 2)); x_n];
    K = size(aug_x_n, 1); % New filter_order
    
    % Initializing variables
    error = zeros(1, N);
    weights = zeros(K, N + 1); % Stores weight time-evolution
    x_hat = zeros(1, N);
    
    for n = 1:N
        s = weights(:, n)' * aug_x_n(:, n);
        x_hat(n) = scale * tanh(s); % Applying scaled tanh activation function
        
        % LMS error
        error(n) = input_x(n) - x_hat(n);
        
        % LMS update rule
        weights(:, n + 1) = weights(:, n) + (mu * scale * (1 - (tanh(s)^2)) * error(n)) * aug_x_n(:, n);
    end
    weights = weights(:, 2:end); % Discarding the first term
end





function [x_hat, error, weights] = pretraining(input_x, mu, filter_order, scale, w_init)

    % Initialization
    N = length(input_x); 
    x_hat = zeros(1, N);
    error = zeros(1, N);
    weights = zeros(filter_order + 1, N + 1); % Include bias term
    
    % Construct augmented input - to take care of bias
    aug_x_n = [ones(1, N); toeplitz([input_x(1), zeros(1, filter_order - 1)], input_x)];
    
    % Using pre-training weights
    weights(:, 1) = w_init;
       
    for n = 1:N
        % Calculate output
        s = weights(:, n)' * aug_x_n(:, n);
        x_hat(n) = scale * tanh(s); % Applying scaled tanh activation function
        
        % Compute error
        error(n) = input_x(n) - x_hat(n);
        
        % Update weights
        weights(:, n + 1) = weights(:, n) + mu * scale * (1 - tanh(s)^2) * error(n) * aug_x_n(:, n);
    end
    
    % Discard the first term
    weights = weights(:, 2:end);
end