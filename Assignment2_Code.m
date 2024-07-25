%% (b)  LMS adaptive predictor for 1 realization

%% Initialization
N =1000; % Number of samples
noise_var = 0.25; % Noise variance
a = [1, -0.1, -0.8]; % AR process coefficients
mu = [0.05, 0.01]; % Step sizes for the LMS algorithm
Num_iter = 1; % Number of realizations for averaging

% Preallocate matrices for error storage
error_matrix_1 = zeros(Num_iter, N); % For mu = 0.05
error_matrix_2 = zeros(Num_iter, N); % For mu = 0.01

%% Process Multiple Realizations
for iter = 1:Num_iter
    % Generate AR process
    h = sqrt(noise_var) * randn(N, 1); % White Gaussian noise
    x = filter(1, a, h); % AR model output
    
    % Initialize errors and weights for each mu
    errors_1 = zeros(N, 1);
    errors_2 = zeros(N, 1);
    weights_1 = zeros(2, N+1); % Including weight at n=0
    weights_2 = zeros(2, N+1); % Including weight at n=0
    
    % LMS Algorithm
    for n = 3:N % Start from 3 since we need x(n-1) and x(n-2)
        % Current input vector
        xn = [x(n-1); x(n-2)];
        
        % Update for mu = 0.05
        x_hat_1 = weights_1(:,n)' * xn;
        errors_1(n) = x(n) - x_hat_1;
        weights_1(:,n+1) = weights_1(:,n) + mu(1) * errors_1(n) * xn;
        
        % Update for mu = 0.01
        x_hat_2 = weights_2(:,n)' * xn;
        errors_2(n) = x(n) - x_hat_2;
        weights_2(:,n+1) = weights_2(:,n) + mu(2) * errors_2(n) * xn;
    end
    
    % Store squared prediction errors
    error_matrix_1(iter, :) = errors_1.^2;
    error_matrix_2(iter, :) = errors_2.^2;
end

% Calculate mean squared error in dB
mse_1_dB = 10 * log10(mean(error_matrix_1, 1));
mse_2_dB = 10 * log10(mean(error_matrix_2, 1));

%% Plotting
figure;

plot(mse_1_dB,  'DisplayName', ['\mu = ', num2str(mu(1))]);
hold on;
plot(mse_2_dB, 'DisplayName', ['\mu = ', num2str(mu(2))]);
title(' Squared Prediction Error vs. Time Step for 1 Realization of x(n)');
xlabel('Time Step');
ylabel('Squared Prediction Error (dB)');
legend show;


%% (b) for 100 realizations

%% Initialization
N =1000; % Number of samples
noise_var = 0.25; % Noise variance
a = [1, -0.1, -0.8]; % AR process coefficients
mu = [0.05, 0.01]; % Step sizes for the LMS algorithm
Num_iter = 100; % Number of realizations for averaging

% Preallocate matrices for error storage
error_matrix_1 = zeros(Num_iter, N); % For mu = 0.05
error_matrix_2 = zeros(Num_iter, N); % For mu = 0.01

%% Process Multiple Realizations
for iter = 1:Num_iter
    % Generate AR process
    h = sqrt(noise_var) * randn(N, 1); % White Gaussian noise
    x = filter(1, a, h); % AR model output
    
    % Initialize errors and weights for each mu
    errors_1 = zeros(N, 1);
    errors_2 = zeros(N, 1);
    weights_1 = zeros(2, N+1); % Including weight at n=0
    weights_2 = zeros(2, N+1); % Including weight at n=0
    
    % LMS Algorithm
    for n = 3:N % Start from 3 since we need x(n-1) and x(n-2)
        % Current input vector
        xn = [x(n-1); x(n-2)];
        
        % Update for mu = 0.05
        x_hat_1 = weights_1(:,n)' * xn;
        errors_1(n) = x(n) - x_hat_1;
        weights_1(:,n+1) = weights_1(:,n) + mu(1) * errors_1(n) * xn;
        
        % Update for mu = 0.01
        x_hat_2 = weights_2(:,n)' * xn;
        errors_2(n) = x(n) - x_hat_2;
        weights_2(:,n+1) = weights_2(:,n) + mu(2) * errors_2(n) * xn;
    end
    
    % Store squared prediction errors
    error_matrix_1(iter, :) = errors_1.^2;
    error_matrix_2(iter, :) = errors_2.^2;
end

% Calculate mean squared error in dB
mse_1_dB = 10 * log10(mean(error_matrix_1, 1));
mse_2_dB = 10 * log10(mean(error_matrix_2, 1));

%% Plotting
figure;

plot(mse_1_dB, 'DisplayName', ['\mu = ', num2str(mu(1))]);
hold on;
plot(mse_2_dB,  'DisplayName', ['\mu = ', num2str(mu(2))]);
title(' Averaged Squared Prediction Error vs. Time Step for 100 Realizations of x(n)');
xlabel('Time Step');
ylabel('Squared Prediction Error (dB)');
legend show;


%% (c) LMS Misadjustments

%Theoretical Misadjustment 

% Given values
R_xx = [0.926 0.463; 0.463 0.926]; % Autocorrelation matrix
sigma_eta_squared = 0.25; % Noise variance
mu_values = [0.05, 0.01]; % Step sizes

% Calculate the trace of R_xx
Tr_R_xx = trace(R_xx);

% Calculate theoretical M_LMS for each mu
M_LMS_theoretical = zeros(size(mu_values));
for i = 1:length(mu_values)
    mu = mu_values(i);
    M_LMS_theoretical(i) = mu * Tr_R_xx / 2;
end

% Display theoretical Misadjustment M_LMS
disp('Theoretical Misadjustment (M_LMS) for each mu:');
disp(['mu = 0.05: ', num2str(M_LMS_theoretical(1))]);
disp(['mu = 0.01: ', num2str(M_LMS_theoretical(2))]);

% Estimated Misadjustment 
% Assuming the error stabilizes in the last 20% of the samples
steady_state_range = 800:N;

% Calculate the steady-state MSE for each mu by averaging over the selected steady-state range
steady_state_mse_1 = mean(mean(error_matrix_1(:, steady_state_range)));
steady_state_mse_2 = mean(mean(error_matrix_2(:, steady_state_range)));

% Convert to dB for consistency with previous plots
steady_state_mse_1 = steady_state_mse_1;
steady_state_mse_2 = steady_state_mse_2;

% Display the steady-state MSE in dB
disp(['Steady-state MSE for mu = 0.05: ', num2str(steady_state_mse_1)]);
disp(['Steady-state MSE for mu = 0.01: ', num2str(steady_state_mse_2)]);


% Calculate EMSE from the MSE values by subtracting sigma_eta_squared
EMSE_mu1 = steady_state_mse_1 - sigma_eta_squared;
EMSE_mu2 = steady_state_mse_2- sigma_eta_squared;

% Calculate experimental M
M_experimental_mu1 = EMSE_mu1 / sigma_eta_squared;
M_experimental_mu2 = EMSE_mu2 / sigma_eta_squared;

% Display experimental Misadjustment M
disp(['Experimental Misadjustment (M) for mu = 0.05: ', num2str(M_experimental_mu1)]);
disp(['Experimental Misadjustment (M) for mu = 0.01: ', num2str(M_experimental_mu2)]);



%% (d) Time evolution of Filter coefficients


% Initialization
N = 1000;
Num_iter = 100;
noise_var = 0.25;
a = [1, -0.1, -0.8];
mu_values = [0.05, 0.01];
num_coef = 2;

% Preallocating arrays for storing coefficients
coefficients = zeros(Num_iter, num_coef, N, length(mu_values));

for mu_idx = 1:length(mu_values)
    mu = mu_values(mu_idx);
    for iter = 1:Num_iter
        % Signal generation
        x = filter(1, a, sqrt(noise_var) * randn(N, 1));

        % Initialize LMS variables
        weights = zeros(num_coef, N + 1);
        % Fixing x_n initialization
        x_n = zeros(num_coef, N);
        for k = 1:num_coef
            x_n(k, :) = [zeros(1, k), x(1:N-k)']; % Ensuring the dimensions match for concatenation
        end

        % LMS Algorithm
        for n = 1:N
            % Making sure to properly access x_n columns
            x_hat = weights(:, n)' * x_n(:, n);
            error = x(n) - x_hat;
            weights(:, n + 1) = weights(:, n) + mu * error * x_n(:, n);
        end

        % Store coefficients from this trial
        coefficients(iter, :, :, mu_idx) = weights(:, 2:end);
    end
end

% Calculating and plotting the mean of coefficients
figure;
for mu_idx = 1:length(mu_values)
    subplot(1, 2, mu_idx);
    mean_coeffs = squeeze(mean(coefficients(:, :, :, mu_idx), 1));
    plot(mean_coeffs');
    hold on;
    yline(-a(2), '--', 'Color', 'r', 'LineWidth', 1.2);
    yline(-a(3), '--', 'Color', 'b', 'LineWidth', 1.2);
    hold off;
    
    title(sprintf('Coefficient Evolution for \\mu=%.2f', mu_values(mu_idx)));
    xlabel('Sample');
    ylabel('Coefficient Value');
    legend('Estimated a_1', 'Estimated a_2', 'True a_1', 'True a_2', 'Location', 'best');
    ylim([-1, 1]);
end


%% (f) - Leaky LMS implementation


%% Parameters
a1 = 0.1; a2 = 0.8; std_eta = 0.5;
N = 1000; % Number of samples after removing the transient
num_trials = 100; % Number of independent trials
order = 2; % Order of the AR process

%% Generate the AR process signal
transient_length = 500; % Length of the transient response
total_length = N + transient_length; % Total length including the transient
data = filter([1], [1, -a1, -a2], std_eta * randn(total_length, num_trials));
data = data(transient_length + 1:end, :); % Remove the transient response

%% Learning rates and leakage coefficients
lrs = [0.05, 0.01];
gammas = [0.01, 0.1, 0.5, 0.9];
params_tot = zeros(order, N, num_trials, length(lrs), length(gammas));

%% Initialize the figure for subplots
figure;
plot_counter = 1; % Counter for subplot indexing

%% Leaky LMS algorithm
for j = 1:length(lrs) % For each learning rate
    for k = 1:length(gammas) % For each gamma value
        for trial = 1:num_trials % For each trial
            % Initialize the weights and the error
            weights = zeros(order, 1);
            errors = zeros(N, 1);
            
            for n = order + 1:N % For each time step
                % Flip the data to make the x vector
                x_vec = flip(data(n-order:n-1, trial));
                
                % Calculate the error
                errors(n) = data(n, trial) - weights' * x_vec;
                
                % Update the weights
                weights = (1 - lrs(j) * gammas(k)) * weights + ...
                          lrs(j) * errors(n) * x_vec;
                
                % Store the weights
                params_tot(:, n, trial, j, k) = weights;
            end
        end
        
        % Average the weights over all trials
        mean_weights = squeeze(mean(params_tot(:, :, :, j, k), 3));
        
        % Plotting
        subplot(length(lrs), length(gammas), plot_counter);
        plot(1:N, mean_weights(1, :), 'LineWidth', 1.2);
        hold on;
        plot(1:N, mean_weights(2, :), 'LineWidth', 1.2);
        yline(a1, '--', 'color', 'r', 'LineWidth', 1.2);
        yline(a2, 'k--', 'LineWidth', 1.2);
        title(sprintf('Leaky LMS - \\mu=%.2f, \\gamma=%.2f', lrs(j), gammas(k)));
        xlabel('Sample');
        ylabel('Coefficient Value');
        legend('Estimate a1', 'Estimate a2', 'True a1', 'True a2');
        hold off;
        ylim([-0.1 1])
        % Update the counter for the next subplot
        plot_counter = plot_counter + 1;
    end
end

%% Adjust the figure size if needed
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 1, 1]); % Maximize the figure window

%% 2.2 (a) 


clear all; clc;

% Initialization
wo = 0.9;
std_eta = sqrt(0.5);
input = std_eta * randn(1500, 1000);
output = filter([1, wo], [1], input);
output = output(501:end, :); % Remove transient effects
input = input(501:end, :);

% Parameters
rho = 0.001;
alpha = 0.8;
lr_fixed = [0.01, 0.1]; % Fixed LMS learning rates
lr_gass_initial = [0.1, 0.2]; % Initial step sizes for GASS

gass = {'standard', 'ben', 'af', 'mx'};

% Preallocate for parameters
params_tot = zeros(4, 1000, 1000); % For 2 fixed LMS and 2 GASS methods
figure;

for lr_index = 1:length(lr_gass_initial)
    lr_gass = lr_gass_initial(lr_index); % Current GASS learning rate
    lrs = [lr_fixed, repmat(lr_gass, 1, length(gass))]; % Combine learning rates

    % Run simulations for each method
    for methodIndex = 1:length(lrs)
        for trial = 1:1000
            % Adjust method selection for GASS or fixed LMS
            method = 'standard'; % Default for fixed LMS
            if methodIndex > length(lr_fixed) % Select GASS method
                method = gass{methodIndex - length(lr_fixed)};
            end
            
            % Perform LMS or GASS algorithm
            [~, params_tot(methodIndex, :, trial), ~] = ...
                lms_arma(output(:, trial), input(:, trial), 0, 1, lrs(methodIndex), method, rho, alpha, 0);
        end

        % Plot results for this method
        subplot(1, 2, lr_index);
        param_error = -(squeeze(mean(params_tot(methodIndex, :, :), 3) - wo));
        hold on;
        plot(param_error, 'LineWidth', 1.5);
    end

    % Configure plot settings
    set(gca, 'fontsize', 18);
    legend('$\mu=0.01$ (Fixed LMS)', '$\mu=0.1$ (Fixed LMS)', 'Benveniste', 'Ang \& Farhang', 'Matthews \& Xie', ...
           'Interpreter', 'Latex', 'Location', 'best');
    title(sprintf('GASS (Initial Step Size = %.1f) and Fixed LMS Weight Error Curves', lr_gass), 'Interpreter', 'Latex');
    ylabel('Weight Error'); 
    xlabel('Sample');
    ylim([0, 1]);
    hold off;
end



%% 2.2(c)


clear all; clc;
% Computing the parameter and error dynamics for each method
wo = 0.9; std_eta = (0.5).^0.5;
input = std_eta*randn(750, 10000);
output = filter([1, wo], [1], input);
output = output(501:end, :); % remove transient filter effects
input = input(501:end, :);

rho_g = 0.005; rho_b = 0.002;
lr_g = 1; lr_b = 0.1;

figure(1);
params_tot_g = zeros(250, 10000); error_tot_g = zeros(250, 10000);
params_tot_b = zeros(250, 10000); error_tot_b = zeros(250, 10000);
for i = 1:10000
    if mod(i, 100) == 0
        sprintf('Realisation %d', i);
    end
    [~, params_tot_g(:, i), error_tot_g(:, i)] = ...
        gngd(output(:, i), input(:, i), 0, 1, lr_g, rho_g, 0);
    [~, params_tot_b(:, i), error_tot_b(:, i)] = ...
        lms_arma(output(:, i), input(:, i), 0, 1, lr_b,'ben', rho_b, 0, 0);
end
% Averaging & Processing
params_tot_g = (squeeze(mean(params_tot_g, 2)));
params_tot_b = (squeeze(mean(params_tot_b, 2)) );
error_tot_g = squeeze(mean(mag2db(error_tot_g.^2), 2));
error_tot_b = squeeze(mean(mag2db(error_tot_b.^2), 2));

% Plot appropriate graphs obtained
figure;
xlim([0 200]); ylim([0 1]); set(gca,'fontsize', 16); hold on;
plot([1:length(params_tot_g)], params_tot_g);
plot([1:length(params_tot_b)], params_tot_b);
title('Weight Estimates for GNGD and Benveniste Algorithm')
legend('GNGD', 'Benveniste');
ylabel('Weight Magnitude'); xlabel('Sample');
hold off;

%% 2.3 (a)

% Define the parameters
N = 1000; % Number of samples
M = 5; % Filter length
muLMS = 0.01; % Learning rate
omega = 0.01 * pi; % Angular frequency of the clean signal

% Generate signals
n = 0:N-1;
x = sin(omega * n); % Clean signal
v = randn(1, N); % White noise with unit variance
eta = v + 0.5 * [zeros(1, 2), v(1:end-2)]; % Colored noise
s = x + eta; % Noise-corrupted signal

% Range of Delta values to test
Delta_values = 1:10;
MSPE = zeros(size(Delta_values)); % To store MSPE for each Delta

% Loop over Delta values and use ALE_LMS function
for Delta = Delta_values
    [w, xhat, error] = ale_lms(s, muLMS, Delta, M);
    % Compute MSPE for the current Delta
    MSPE(Delta) = mean((x(Delta+M:end) - xhat(Delta+M:end)).^2);
end

% Plotting the MSPE for different Delta values
figure;
plot(Delta_values, MSPE, '-o');
xlabel('\Delta');
ylabel('MSPE');
title('MSPE for Different \Delta Values in ALE for M = 5');
grid on;

%% 2.3 (b)b(i)


% Define the parameters
N = 1000; % Number of samples
muLMS = 0.01; % Learning rate
omega = 0.01 * pi; % Angular frequency of the clean signal
M_values = [5, 10, 15, 20]; % Filter lengths to test
Delta_values_full = 3:25; % Full range of Delta values to test

% Generate signals
n = 0:N-1;
x = sin(omega * n); % Clean signal
v = randn(1, N); % White noise with unit variance
eta = v + 0.5 * [zeros(1, 2), v(1:end-2)]; % Colored noise
s = x + eta; % Noise-corrupted signal

MSPE_matrix = zeros(length(M_values), length(Delta_values_full));

for mIdx = 1:length(M_values)
    M = M_values(mIdx);
    for dIdx = 1:length(Delta_values_full)
        Delta = Delta_values_full(dIdx);
        [~, xhat, error] = ale_lms(s, muLMS, Delta, M);
        MSPE_matrix(mIdx, dIdx) = mean((x(Delta+M:end) - xhat(Delta+M:end)).^2);
    end
end


figure;
subplot(1, 3, 1); % Use subplot to arrange plots horizontally
hold on; % Allows multiple lines to be plotted on the same figure
colors = lines(length(M_values)); % Obtain distinct colors for each line
for mIdx = 1:length(M_values)
    plot(Delta_values_full, MSPE_matrix(mIdx, :), '-o', 'Color', colors(mIdx,:), 'DisplayName', ['M = ', num2str(M_values(mIdx))]);
end
hold off;
xlabel('\Delta');
ylabel('MSPE');
title('MSPE for Different \Delta Values Across M');
legend show;
grid on;

% Specific Delta values for overlay plots
Delta_specific = [3, 25];
M_specific = 5; % Specific filter order for overlay plots

% Loop for subplots 2 and 3: Overlay plots for specific Delta values and M=5
for i = 1:2
    Delta = Delta_specific(i);
    subplot(1, 3, i+1); % Arrange these plots horizontally
    [~, xhat, ~] = ale_lms(s, muLMS, Delta, M_specific);
   
    plot(n, s, ':', 'LineWidth', 1);hold on;
   
     plot(n, xhat, '-.', 'LineWidth', 2);
      plot(n, x, 'b', 'LineWidth', 2);
    hold off;
    legend( 'Noise-Corrupted Signal','Estimated Signal', 'Clean Signal', 'Location', 'best');
    xlabel('Sample');
    ylabel('Signal Amplitude');
    title(['\Delta = ', num2str(Delta), ', M = ', num2str(M_specific)]);
    grid on;
end

%% 2.3 (c)


%Number of samples to use
N=1000;

%Generating clean signal
omega_0 = 0.01*pi;
n=(0:N-1)';
x = sin(omega_0.*n);

%Filter coefficients for coloured noise
a=1;
b=[1 0 0.5];


mu=0.01; %LMS learning rate
filter_order = 5; %LMS filter order
N_iter=100; %Number of iterations
x_approx_3= zeros(N_iter,N);
x_approx_4 = zeros(N_iter,N);
MSPE_3 = zeros(N_iter,1);
MSPE_4 = zeros(N_iter,1);
delay = 3; %minimum delay used
figure;
for iter = 1:N_iter
    v = randn(N,1); %white noise
    h=filter(b,a,v); %obtaining coloured noise
    s = x+h; %Noise-corrupted signal
    u=2*h - 0.3;%Obtaining reference signal
    
    %Computing ALE algorithm
    [x_hat,~,~] = ALE_lms(s,mu,filter_order,delay);
    x_approx_3(iter,:)=x_hat;
    MSPE_3(iter) = mean((x-x_hat).^2);
    
    %Computing ANC algorithm
    [noise_est,x_hat,~] = ANC_lms(u,s,mu,filter_order);
    x_approx_4(iter,:)=x_hat;
    MSPE_4(iter) = mean((x-x_hat).^2);
    
end
%Obtaining the ensemble-averaged signal
x_approx_ALE = mean(x_approx_3);
x_approx_ANC = mean(x_approx_4);

subplot(1,2,1)
plot(x_approx_ALE,'Linewidth',1)
hold on
plot(x,'k','Linewidth',1)
grid on
xlabel('Sample','FontSize',11); ylabel('Signal Amplitude','FontSize',11);
legend('ALE estimate','Clean')
MSPE_ALE = mean(MSPE_3);
title(['ALE with MSE = ', num2str(MSPE_ALE),' | M=5 and \Delta=3'])



subplot(1,2,2)
plot(x_approx_ANC,'Linewidth',1)
hold on
plot(x,'k','Linewidth',1)
grid on
xlabel('Sample','FontSize',11); ylabel('Signal Amplitude','FontSize',11);
legend('ANC estimate','Clean')
MSPE_ANC = mean(MSPE_4);
title(['ANC with MSE = ', num2str(MSPE_ANC),' | M=5'])

%% 2.3 (d)

clear all;
clc
load 'EEG_Data_Assignment2.mat';
c = Cz;
Cz_norm = normalize(c);
data = detrend(Cz);

%% Constructing time axis
dt=1/fs; %In seconds
stop_time=length(data)*dt;
t = (0:dt:stop_time-dt)'; % seconds.

%% Constructing synthetic reference input
f0 = 50; % Sine wave frequency (hertz)
ref_signal = sin(2*pi*f0*t) + 0.001*randn(size(t));

% spectrogram parameters
L=4*fs; %Length of windows
perc_over = 0.5; %percentage overlap
nOverlap = round(perc_over * L); %samples overlapping
nfft = 3*L; % nFFT points

%% Noise-corrupted EEG data
figure;
spectrogram(data, hanning(L),nOverlap , nfft, fs, 'yaxis');
ylim([0 100])
title('Spectrogram of noise-corrupted EEG data (Cz)')

% Varying parameters

mu=[0.001,0.01,0.1];
%Model order
M=[3,10,20];
index=1;
for order_ind =1:length(M)
    for mu_ind=1:length(mu)
        %Computing ANC algorithm
        [noise_est,x_hat,~] = ANC_lms(ref_signal,data,mu(mu_ind),M(order_ind));
        subplot(3,3,index)
        spectrogram(x_hat, hanning(L),nOverlap , nfft, fs, 'yaxis');
        title(['M = ',num2str(M(order_ind)),' and \mu = ', num2str(mu(mu_ind))]);
        ylim([0 100]);
        index=index+1;
    end
end

mu_opt = 0.001;
M_opt = 10;

%Computing ANC algorithm
[noise_est,x_hat,~] = ANC_lms(ref_signal,data,mu_opt,M_opt);figure;

spectrogram(x_hat, hanning(L),nOverlap , nfft, fs, 'yaxis');
title(['De-noised EEG signal (M=',num2str(M_opt),',\mu=',num2str(mu_opt),')']);
ylim([0 100])





%%
function [ar_params, ma_params, error] = lms_arma(output, input, p, q, lr, gass, rho, alpha, leak)

params = zeros(p+q, length(output)); % Combined AR and MA parameters
phi = zeros(p+q, length(output)); % Auxiliary vector for GASS update
lrs = lr * ones(size(output)); % Learning rates
error = ones(size(output)); % Prediction errors

for i = max([p,q]) + 1 : length(output) - 1
    % Construct augmented data vector
    aug_dat = [flip(output(i-p:i-1)); flip(input(i-q:i-1))]';
    % Calculate prediction error
    error(i) = output(i) - dot(aug_dat, params(:, i));
    % Parameter update with leakage
    params(:, i+1) = (1 - leak * lr) * params(:, i) + lrs(i) * error(i) * aug_dat;
    % Update step-size according to selected GASS method
    if i > max([p,q]) + 1
        switch gass
            case 'af'
                phi(:, i) = alpha * phi(:, i-1) + error(i-1) * prev_aug_dat;
            case 'ben'
                phi(:, i) = (eye(length(prev_aug_dat)) - lrs(i-1) * (prev_aug_dat * prev_aug_dat')) * phi(:, i-1) + error(i-1) * prev_aug_dat;
            case 'mx'
                phi(:, i) = error(i-1) * prev_aug_dat;
        end
        if ~strcmp(gass, 'standard')
            lrs(i+1) = lrs(i) + rho * error(i) * dot(aug_dat, phi(:, i));
        end
    end
    prev_aug_dat = aug_dat; % Save current augmented data for the next iteration
end

% Extract AR and MA parameters from combined parameter matrix
ar_params = params(1:p, :); 
ma_params = params(p+1:end, :);

end






function [ar_params, ma_params, error] = gngd(output, input, p, q, lr, rho, leak)
% Initialize parameters and variables
    params = zeros(p+q, length(output));
    error = ones(size(output));
    reg = ones(size(output)) / lr;
    old_dat = zeros(max([p, q]), 1); % Placeholder for previous data vector

    % Main loop for adaptive filtering
    for i = max([p, q]) + 1 : length(output) - 1
        % Constructing augmented data vector from past output and input values
        aug_dat = [flip(output(i-p:i-1)); flip(input(i-q:i-1))];
        % Calculating prediction error
        error(i) = output(i) - aug_dat' * params(:, i);
        % Learning rate adjustment
        lr_now = lr / (reg(i) + aug_dat' * aug_dat);
        % Parameters update with leakage correction
        params(:, i+1) = (1 - leak * lr) * params(:, i) + lr_now * error(i) * aug_dat;
        
        % Regularization parameter update, avoiding this computation on the first iteration
        if i > max([p, q]) + 1
            num = rho * lr * error(i) * error(i-1) * dot(old_dat, aug_dat);
            den = (reg(i-1) + dot(old_dat, old_dat))^2;
            reg(i+1) = reg(i) - num / den;
        end
        old_dat = aug_dat;
    end
    
    % Extract AR and MA parameters from the composite parameters matrix
    ar_params = params(1:p, :);
    ma_params = params(p+1:end, :);
end





function [w, xhat, error] = ale_lms(signal, lr, delta, M)
   
    w = zeros(M, length(signal)); % Adjusted size
    error = zeros(size(signal));
    xhat = zeros(size(signal)); % Ensure xhat is initialized to zeros

    for n = delta+M:length(signal)
        u = flip(signal(n-delta-M+1:n-delta));
        xhat(n) = dot(w(:, n - delta - M + 1), u); % Corrected indexing for w
        error(n) = signal(n) - xhat(n);
       w(:, n - delta - M + 1 + 1) = w(:, n - delta - M + 1) + lr * error(n) * u';

    end
    w = w(:, 2:end); % Exclude the initial column of zeros
end





function [noise_Est,x_hat,weights] = ANC_lms(u,s,mu,order)

N = length(s); % Total number of samples in the signal
% Initialization
noise_Est = zeros(N,1); % Estimated noise
x_hat = zeros(N,1); % Denoised signal
weights = zeros(order, N); % Adaptive filter weights over time

% Padding the reference signal (u) with zeros to handle the filter order
paddedU = [zeros(order-1, 1); u];

for n = 1:N
    % Collecting the current and past order-1 samples from u
    u_n = flipud(paddedU(n:n+order-1));
    
    % Estimate the current noise sample
    noise_Est(n) = weights(:,n)' * u_n;
    
    % Subtracting the estimated noise from the primary input (s) to get the denoised signal
    x_hat(n) = s(n) - noise_Est(n);
    
    % Updating the weights for the next iteration using the LMS algorithm
    if n < N % To ensure we don't go beyond N when indexing
        weights(:, n+1) = weights(:, n) + mu * x_hat(n) * u_n;
    end
end

% Adjusting the weights matrix to exclude the initial zero column
weights = weights(:, 2:end);

end




function [x_hat,error,weights] = ALE_lms(x,mu,order,D)


N = length(x); % Total number of samples
% Initialization
x_hat = zeros(N,1); % Estimated signal
error = zeros(N,1); % Error signal
weights = zeros(order, N); % Adaptive filter weights

% Padding x with zeros at the beginning to account for delay and order
paddedX = [zeros(order+D-1, 1); x];

for n = 1:N
    % Create a vector of current sample and (order-1) previous samples
    x_n = flipud(paddedX(n:n+order-1));
    
    % Estimate the current sample
    x_hat(n) = weights(:,n)' * x_n;
    
    % Calculate error between actual and estimated sample
    error(n) = paddedX(n+order+D-1) - x_hat(n);
    
    % Update the weights using the LMS algorithm
    if n < N % To prevent index exceeding matrix dimensions
        weights(:, n+1) = weights(:, n) + mu * error(n) * x_n;
    end
end

% Adjust weights matrix to exclude the first column which is all zeros
weights = weights(:, 2:end);

end