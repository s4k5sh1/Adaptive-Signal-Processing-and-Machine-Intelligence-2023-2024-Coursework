%% 3.1 (a)

%Signal length
N=100000;
%Generating circular white noise
x = randn(1,N)+1j*randn(1,N); 

%Generating WLMA(1) process
signal = zeros(1,length(x));
b = [1 (1.5+1*1j) (2.5-0.5*1j)];
signal(1) = x(1);
for i = 1:N-1
    signal(i+1) = b(1)*x(i+1) + b(2)*x(i) + b(3)*conj(x(i));
end

% Calculate the Circularity coefficients
circularity_coef_WGN = abs(mean(x .^ 2)) / mean(abs(x) .^ 2);
circularity_coef_signal = abs(mean(signal .^ 2)) / mean(abs(signal) .^ 2);

%Checking circularity of WGN
figure;
subplot(1,2,1)
scatter(real(x),imag(x),'.')
xlabel('Real part')
ylabel('Imaginary part')
grid on

axis equal; 
title(['WGN with |\rho| =',num2str(round(circularity_coef_WGN,4))]) 

%Checking circularity of Signal (WLMA(1) process)
subplot(1,2,2)
scatter(real(signal),imag(signal),'r','.')
xlabel('Real part')
ylabel('Imaginary part')
grid on
axis equal;
title(['WLMA(1) with |\rho| =',num2str(round(circularity_coef_signal,3))]) 
N_iter=100;
mu = 0.03;  %learning rate
num_coef=length(b)-1;
N=1000; %Signal length

%Initialising matrices
error_CLMS = zeros(N_iter,N);
error_ACLMS = zeros(N_iter,N);

for iter=1:N_iter
    %Generating circular white noise
    x = randn(1,N)+1j*randn(1,N);
    %Generating WLMA(1) process
    signal = zeros(1,length(x));
    b = [1 (1.5+1*1j) (2.5-0.5*1j)];
    signal(1) = x(1);
    for i = 1:N-1
        signal(i+1) = b(1)*x(i+1) + b(2)*x(i) + b(3)*conj(x(i));
    end
        
    % Implementing the CLMS algorithm
    [signal_est_CLMS,error_1,~] = CLMS_MA(x,signal,mu,num_coef);
    error_CLMS(iter,:) = error_1;
    
    %Implementing the ACLMS algorithm
    [signal_est_ACLMS,error_2,~,~] = ACLMS_MA(x,signal,mu,num_coef);
    error_ACLMS(iter,:) = error_2;
end

%Obtaining learning curves
figure;

plot(10 * log10(mean(abs(error_CLMS).^2,1)),'Linewidth',1)
hold on
plot(10 * log10(mean(abs(error_ACLMS).^2,1)),'Linewidth',1)
xlabel('Sample','FontSize',11);
ylabel('Square error (dB)','FontSize',11);
legend('CLMS','ACLMS')
title('Learning curves for CLMS and ACLMS algorithms','FontSize',11)
grid on


%% 3.1 (b)

clc; clear variables; close all;

% Load the wind data from MAT files
high_wind = load('high-wind.mat');
medium_wind = load('medium-wind.mat');
low_wind = load('low-wind.mat');

% Assuming the MAT files contain variables named 'v_east' and 'v_north',
% and these variables are directly accessible after loading.
% Form complex-valued wind signals
high_wind_complex = high_wind.v_east + 1j * high_wind.v_north;
medium_wind_complex = medium_wind.v_east + 1j * medium_wind.v_north;
low_wind_complex = low_wind.v_east + 1j * low_wind.v_north;

% Collect the wind signals into an array or cell array for easier processing
wind_signals = {high_wind_complex, medium_wind_complex, low_wind_complex};

% Correct colors and titles to match provided instruction
colors = {'r', 'b', 'm'}; % Red, Blue, Magenta for high, medium, and low speeds respectively
titles = {'High-speed wind data', 'Medium-speed wind data', 'Low-speed wind data'};

mus = [0.001, 0.01, 0.1]; % Learning rates for high, medium, low wind speeds
filter_orders = 1:30; % Filter lengths to test
mse_CLMS = zeros(length(filter_orders), 3); % MSE storage for CLMS
mse_ACLMS = zeros(length(filter_orders), 3); % MSE storage for ACLMS

% Plot circularity plots and compute MSE
figure;
for i = 1:3 % Loop over wind speeds
    % Plotting circularity plots
    subplot(1, 3, i);
    scatter(real(wind_signals{i}), imag(wind_signals{i}), '.', 'MarkerEdgeColor', colors{i});
    title([titles{i} ' with |ρ| = ' num2str(round(abs(mean(wind_signals{i} .^ 2)) / mean(abs(wind_signals{i}) .^ 2), 4))]);
    xlabel('Real part (East)');
    ylabel('Imaginary part (North)');
    grid on; axis equal;
    
    % MSE computation
    x_signal = [0; wind_signals{i}(1:end-1)]; % Lagged signal for prediction
    d_signal = wind_signals{i}; % Desired signal
    for j = filter_orders
        [~, error_CLMS] = CLMS_wind(x_signal, d_signal, mus(i), j);
        mse_CLMS(j, i) = mean(abs(error_CLMS).^2);
        
        [~, error_ACLMS] = ACLMS_wind(x_signal, d_signal, mus(i), j);
        mse_ACLMS(j, i) = mean(abs(error_ACLMS).^2);
    end
end

% Plot MSE results in dB
figure;
for i = 1:3
    subplot(1, 3, i);
    plot(filter_orders, pow2db(mse_CLMS(:, i)), 'LineWidth', 1); hold on;
    plot(filter_orders, pow2db(mse_ACLMS(:, i)), 'LineWidth', 1);
    title([titles{i} '-speed Wind Data']);
    xlabel('Model Order');
    ylabel('MSPE (dB)');
    legend('CLMS', 'ACLMS'); grid on;
end

%% 3.1 (c)

% Define system parameters
fs = 1000;  % Sampling frequency (Hz)
f0 = 50;    % System frequency (Hz)
phi = 0;    % Phase shift (radians)
V = 1;      % Peak voltage (balanced condition)
N = 1000;   % Number of samples
t = (0:N-1)'/fs;  % Time vector

% Generate balanced voltages
va_bal = V * cos(2*pi*f0*t + phi);
vb_bal = V * cos(2*pi*f0*t + phi + 2*pi/3);
vc_bal = V * cos(2*pi*f0*t + phi - 2*pi/3);

% Generate unbalanced voltages (magnitude only change)
va_unbal_mag = 1 * cos(2*pi*f0*t + phi);  % Slightly higher magnitude
vb_unbal_mag = 1.1 * cos(2*pi*f0*t + phi + 2*pi/3);
vc_unbal_mag = 0.5 * cos(2*pi*f0*t + phi - 2*pi/3);

% Generate unbalanced voltages (phase only change)
va_unbal_phase = V * cos(2*pi*f0*t + phi + 30);  % Slightly phase shifted
vb_unbal_phase = V * cos(2*pi*f0*t + phi + 2*pi/3 );
vc_unbal_phase = V * cos(2*pi*f0*t + phi - 2*pi/3 - 0.1);

% Clarke Transformation matrix
C = sqrt(2/3) * [sqrt(2)/2 sqrt(2)/2 sqrt(2)/2; 1 -1/2 -1/2; 0 sqrt(3)/2 -sqrt(3)/2];

% Apply Clarke Transform to balanced and unbalanced systems (magnitude only)
v_bal_mag = C * [va_bal vb_bal vc_bal]';
v_unbal_mag = C * [va_unbal_mag vb_unbal_mag vc_unbal_mag]';

% Apply Clarke Transform to balanced and unbalanced systems (phase only)
v_bal_phase = C * [va_bal vb_bal vc_bal]';
v_unbal_phase = C * [va_unbal_phase vb_unbal_phase vc_unbal_phase]';

% Extract alpha-beta components for plotting
v_alpha_beta_bal_mag = v_bal_mag(2,:) + 1j*v_bal_mag(3,:);
v_alpha_beta_unbal_mag = v_unbal_mag(2,:) + 1j*v_unbal_mag(3,:);

v_alpha_beta_bal_phase = v_bal_phase(2,:) + 1j*v_bal_phase(3,:);
v_alpha_beta_unbal_phase = v_unbal_phase(2,:) + 1j*v_unbal_phase(3,:);

% Calculate circularity coefficients for balanced and unbalanced systems
% Balanced system
circ_coef_bal = abs(mean(v_alpha_beta_bal_mag.^2)) / mean(abs(v_alpha_beta_bal_mag).^2);
% Unbalanced system due to magnitude only change
circ_coef_unbal_mag = abs(mean(v_alpha_beta_unbal_mag.^2)) / mean(abs(v_alpha_beta_unbal_mag).^2);
% Unbalanced system due to phase only change
circ_coef_unbal_phase = abs(mean(v_alpha_beta_unbal_phase.^2)) / mean(abs(v_alpha_beta_unbal_phase).^2);

% Plot circularity diagrams
figure;

% Magnitude Change Circularity Diagram
subplot(1, 2, 1);
grid on;
plot(real(v_alpha_beta_bal_mag), imag(v_alpha_beta_bal_mag),LineWidth=1.5);
hold on;
plot(real(v_alpha_beta_unbal_mag), imag(v_alpha_beta_unbal_mag),LineWidth=1.5);
axis equal;
xlabel('Real Part');
ylabel('Imaginary Part');
title({'Balanced vs Unbalanced (Magnitude Only)', ...
       sprintf('Balanced |\\rho| = %.4f, Unbalanced |\\rho| = %.4f', circ_coef_bal, circ_coef_unbal_mag), ...
       sprintf('Magnitudes: V_{a} = %g, V_{b} = %g, V_{c} = %g', V, 1.1*V, 0.5*V)});
legend({'Balanced', 'Unbalanced (Magnitude Only)'});
xlim([-2 2])
% Phase Change Circularity Diagram
subplot(1, 2, 2);
grid on;
plot(real(v_alpha_beta_bal_phase), imag(v_alpha_beta_bal_phase),LineWidth=1.5);
hold on;
plot(real(v_alpha_beta_unbal_phase), imag(v_alpha_beta_unbal_phase),LineWidth=1.5);
axis equal;
xlabel('Real Part');
ylabel('Imaginary Part');
title({'Balanced vs Unbalanced (Phase Only)', ...
       sprintf('Balanced |\\rho| = %.4f, Unbalanced |\\rho| = %.4f', circ_coef_bal, circ_coef_unbal_phase), ...
       sprintf('Phase Shift: |\\Delta_{a}| = %g rad, |\\Delta_{c}| = %g rad', 30, -0.1)});
legend({'Balanced', 'Unbalanced (Phase Only)'});
xlim([-2 2])

%% 3.1 (e)


% Define simulation parameters
N = 5000; % Number of time samples
n = 0:N-1; % Time axis
f_0 = 50; % System frequency in Hz
f_s = 1000; % Sampling frequency in Hz
mu = 0.05; % Adaptation gain for both algorithms

% Generate balanced and unbalanced system voltages
V_bal = [1, 1, 1];
delta_bal = [0, 0];
phi_bal = 0; % Common nominal phase shift for balanced system 

V_unbal = [1.8, 0.8, 2.2];
delta_unbal = [1.3,0.5, 2.9];
phi_unbal = 0; % Common nominal phase shift for unbalanced system

% Generating phase voltages and applying Clarke Transform
Voltage_Clarke_bal = generateClarkeVoltage(V_bal, delta_bal, phi_bal, f_0, f_s, n);
Voltage_Clarke_unbal = generateClarkeVoltage(V_unbal, delta_unbal, phi_unbal, f_0, f_s, n);

% Estimate frequencies using CLMS and ACLMS
[f0_bal_CLMS, f0_bal_ACLMS] = estimateFrequency(Voltage_Clarke_bal, f_s, mu);
[f0_unbal_CLMS, f0_unbal_ACLMS] = estimateFrequency(Voltage_Clarke_unbal, f_s, mu);

% Plotting results for balanced and unbalanced systems
plotFrequencyEstimates(f0_bal_CLMS, f0_bal_ACLMS, f0_unbal_CLMS, f0_unbal_ACLMS);

%% 3.2 (a)

clear all; clc;

n_samples = 1500;
f = [100*ones(500,1); ...
    100*ones(500,1) + ([501:1000]' - 500)/2; ...
    100*ones(500,1) + (([1001:1500]' -  1000)/25).^2];
phi = cumsum(f);
fs = 1500;
variance = 0.05;
eta = wgn(n_samples,1,pow2db(variance),'complex');
y = exp(1i*2*pi*phi/fs) + eta;

% Plotting frequency
figure(1); 
subplot(1,3,1); hold on; set(gca,'fontsize', 16);
plot([1:n_samples], f,'K');
grid on;
xlabel('Samples'); ylabel('Frequency (Hz)');
title('Frequency modulated (FM) signal'); hold off;

% Plotting only AR(1) power spectrum in the middle subplot
subplot(1,3,2); hold on; set(gca,'fontsize', 16);
% Finding AR(1) coefficients
a_ar1 = aryule(y, 1);
[H_ar1,w_ar1] = freqz(1,a_ar1,n_samples,fs);
P_ar1 = abs(H_ar1).^2;
plot(w_ar1, pow2db(P_ar1));
grid on;
title('Power Spectrum of FM signal AR(1) model')
xlabel('Frequency (Hz)'); 
ylabel('Magnitude (dB)');
legend('p=1');
hold off;

% Plotting PSD of signal from estimates for different orders
order = [1, 5, 10, 20];
subplot(1,3,3); hold on; set(gca,'fontsize', 16);
H = zeros(4,1500); w = zeros(4,1500);
for i = 1:4
    % Finding AR coefficients
    a = aryule(y,order(i));
    [H(i,:),w(i,:)] = freqz(1,a,n_samples,fs);
    P = abs(H(i,:)).^2;
    plot(w(i,:), pow2db(P));
end
plot(mean(f)*ones(1,1500), linspace(-10,30,1500),'--')
grid on;
title('Power Spectrum of FM signal with Different Model Orders (p)')
xlabel('Frequency (Hz)'); 
ylabel('Magnitude (dB)');
leg = arrayfun(@(a)strcat('p=',num2str(a)),order,'uni',0);
leg{end+1} = 'Average FM Value';
legend(leg); 
hold off;

%% 3.2 (b)

%% General Parameters
N = 1500; % Number of time samples to be used
fs = 1500; % Sampling frequency in Hz

%% Signal Generation
% Generating circular white noise
eta = sqrt(0.05) .* randn(1, N) + 1j * sqrt(0.05) .* randn(1, N);

% Generating phase and frequency directly in the script
freq = [100 * ones(1, 500), 100 + ((1:500) - 1) / 2, 100 + (((1:500) - 1) / 25).^2];
phase = cumsum(freq); % Cumulative sum to get phase

% Generating frequency-modulated (FM) signal
y = exp(1j * ((2 * pi * phase) / fs)) + eta;

%% Implementing the CLMS Algorithm - Data is Mostly Circular
coef = 1; % AR(1) process
N_points = 1024;
figure;
mu_values = [0.001, 0.01, 0.1];

for index = 1:length(mu_values)
    mu = mu_values(index);
    % Estimate AR coefficients using CLMS
    [~, ~, ar_CLMS] = CLMS_AR_3b(y, mu, coef);

    % Initialize matrix to store power spectra
    H = zeros(N_points, N);
    for k = 1:N
        % Compute power spectrum at each time step
        [h, f] = freqz(1, [1; -conj(ar_CLMS(k))'], N_points, fs);
        H(:, k) = abs(h).^2;
    end

    % Normalize power spectra to mitigate outliers
    medianH = 50 * median(median(H));
    H(H > medianH) = medianH;

    % Plot time-frequency diagram
    subplot(1, 3, index)
    surf(1:N, f, H, 'LineStyle', 'none');
    view(2);
    colorbar;
    ylabel('Frequency (Hz)');
    xlabel('Sample Index');
    title(['\mu = ', num2str(mu)]);
    ylim([0 fs/2]); % Adjusted frequency axis to half of sampling rate
end




%% 3.2 (c)

clc; 
clear variables; 
close all;

% Parameters
n_samples = 1500;
fs = 1000; % Sampling frequency
variance = 0.05;

% Generate FM signal
t = (1:n_samples)'; % Time vector
f1 = 100 * ones(500, 1);
f2 = f1 + (t(501:1000) - 500) / 2;
f3 = f1 + ((t(1001:1500) - 1000) / 25).^2;
frequencies = [f1; f2; f3]; % Combined frequency components
phi = cumsum(frequencies); % Phase
eta = wgn(n_samples, 1, pow2db(variance), 'complex'); % Noise
y = exp(1i * 2 * pi * phi / fs) + eta; % FM signal with noise

% Create Fourier input signal
frequencies_axis = (0:fs-1)';
x = (1/fs) * exp(1i * 2 * pi * frequencies_axis * t' ./ fs); % Fourier basis

% Test for different leakages and plot results
leaks = [0, 0.05, 0.5];
figure(1);
for idx = 1:numel(leaks)
    [weights, ~] = dft_clms(y.', x, leaks(idx));
    
    % Plotting setup
    subplot(numel(leaks), 1, idx); hold on; set(gca, 'fontsize', 16);
    mesh(t, frequencies_axis(1:floor(fs/2)), abs(weights(1:floor(fs/2),:)).^2);
    view(2);
    xlabel('Samples'); ylabel('Frequency (Hz)'); ylim([0, 500]);
    title(sprintf('Time-frequency spectrum of FM Signal using DFT-CLMS (\\gamma = %g)', leaks(idx)));
    colorbar; hold off;
end

%% 3.2 (d)


clc; clear variables; close all;
load('EEG_Data_Assignment2.mat');

a = 1000;
t_range = a:a+1199; % Defining the time range
POz = detrend(POz(t_range)); % Detrending the signal within the specified range
n_samples = length(t_range);
fs = 1000; % Sampling frequency, assuming it's defined in 'EEG_Data_Assignment2.mat'
f_ax = 0:fs-1; % Frequency axis

% Correcting the generation of the Fourier input signal
% Using a time vector that corresponds to the points in t_range
time_vec = (0:n_samples-1) / fs; % Time vector in seconds
x = (1/fs) * exp(1i * 2 * pi * f_ax' * time_vec); % Corrected matrix dimensions

[w,~] = dft_clms(POz, x, 0);

% Plotting results
figure(1); hold on; set(gca,'fontsize', 16);
mesh(t_range, f_ax(1:floor(fs/2)), abs(w(1:floor(fs/2),:)).^2);
view(2); ylim([0,100]);
xlabel('Samples');
ylabel('Frequency (Hz)');
title('Time-frequency spectrum of EEG (POz) Signal using DFT-CLMS');
colorbar; % Adds a color bar to indicate magnitude
hold off;


 
%%
function [y, error, weights] = CLMS_MA(x, d, mu, filter_order)
    N = length(d);  % Total number of samples
    weights = zeros(filter_order, N + 1);  % Include N+1 for initial weights
    y = zeros(N, 1);  % Initialize output signal
    error = zeros(N, 1);  % Initialize error signal

    for n = filter_order:N
        x_n = x(n:-1:n-filter_order+1);  % Input vector
        % Ensure x_n is a column vector explicitly
        x_n = reshape(x_n, [filter_order, 1]);
        % Calculate output
        y(n) = weights(:, n)' * x_n;  % Error here indicates x_n might not be correctly shaped
        error(n) = d(n) - y(n);  % Calculate error
        % Update weights
        weights(:, n + 1) = weights(:, n) + mu * conj(error(n)) * x_n;
    end

    weights = weights(:, 2:end);  % Adjust to remove initial zero column
end




function [y, error, h, g] = ACLMS_MA(x, d, mu, filter_order)
    

    N = length(d);  % Number of samples in desired signal
    h = zeros(filter_order, N);  % Weight matrix h initialization
    g = zeros(filter_order, N);  % Weight matrix g initialization
    y = zeros(N, 1);  % Output signal initialization
    error = zeros(N, 1);  % Error signal initialization

    for n = filter_order:N
        % Augmented input vector (using past values and their conjugates)
        x_n = x(n:-1:n-filter_order+1).';
        x_n_conj = conj(x_n);
        
        % Output and error calculation
        y(n) = h(:, n)' * x_n + g(:, n)' * x_n_conj;
        error(n) = d(n) - y(n);
        
        % Weights update
        h(:, n + 1) = h(:, n) + mu * conj(error(n)) * x_n;
        g(:, n + 1) = g(:, n) + mu * conj(error(n)) * x_n_conj;
    end

end


function [y, error, weights] = CLMS_wind(x, d, mu, filter_order)


    N = length(d); % Number of samples
    y = zeros(N, 1); % Initialize output signal
    error = zeros(N, 1); % Initialize error vector with correct size
    weights = zeros(filter_order, N+1); % Initialize weights; include initial condition

    for n = filter_order:N
        % Select the current window of input samples
        x_n = x(n:-1:n-filter_order+1);
        
        % Predict the current sample
        y(n) = weights(:, n)' * x_n;
        
        % Calculate the prediction error
        error(n) = d(n) - y(n);
        
        % Update the weights
        weights(:, n+1) = weights(:, n) + mu * conj(error(n)) * x_n;
    end

    % Discard the initial condition from weights for the output
    weights = weights(:, 2:end);
end




function [y, error, h, g] = ACLMS_wind(x, d, mu, filter_order)


    N = length(d); % Number of samples
    y = zeros(N, 1); % Initialize output signal
    error = zeros(N, 1); % Initialize error vector with the correct size
    h = zeros(filter_order, N+1); % Initialize 'h' weights; include initial condition
    g = zeros(filter_order, N+1); % Initialize 'g' weights for conjugate inputs; include initial condition

    for n = filter_order:N
        % Form the input vector and its conjugate from the last 'filter_order' samples
        x_n = x(n:-1:n-filter_order+1);
        x_conj_n = conj(x_n);
        
        % Calculate the output and error for current sample
        y(n) = h(:, n)' * x_n + g(:, n)' * x_conj_n;
        error(n) = d(n) - y(n);
        
        % Update weights
        h(:, n+1) = h(:, n) + mu * conj(error(n)) * x_n;
        g(:, n+1) = g(:, n) + mu * conj(error(n)) * x_conj_n;
    end

    % Discard the initial condition from weights for the output
    h = h(:, 2:end);
    g = g(:, 2:end);
end




function Voltage_Clarke = generateClarkeVoltage(V, delta, phi, f_0, f_s, n)
    % Generate phase voltages
    V_a = V(1) * cos(2*pi*(f_0/f_s)*n + phi);
    V_b = V(2) * cos(2*pi*(f_0/f_s)*n + phi + delta(1) - (2*pi/3));
    V_c = V(3) * cos(2*pi*(f_0/f_s)*n + phi + delta(2) + (2*pi/3));

    % Apply Clarke Transform
    C = sqrt(2/3) * [sqrt(2)/2, sqrt(2)/2, sqrt(2)/2; 1, -0.5, -0.5; 0, sqrt(3)/2, -sqrt(3)/2];
    Voltage_Clarke = C * [V_a; V_b; V_c];
    Voltage_Clarke = Voltage_Clarke(2,:) + 1j*Voltage_Clarke(3,:);
end





function [f0_CLMS, f0_ACLMS] = estimateFrequency(input_x, f_s, mu)
    coef = 1; % Assuming AR(1) process
    [~, ~, h_CLMS] = CLMS_pred_AR(input_x, mu, coef);
    f0_CLMS = abs((f_s/(2*pi)) * atan(imag(h_CLMS)./real(h_CLMS)));

    [~, ~, h_ACLMS, g_ACLMS] = ACLMS_pred_AR(input_x, mu, coef);
    f0_ACLMS = abs((f_s/(2*pi)) * atan(sqrt((imag(h_ACLMS).^2) - abs(g_ACLMS).^2) ./ real(h_ACLMS)));
end




function plotFrequencyEstimates(f0_bal_CLMS, f0_bal_ACLMS, f0_unbal_CLMS, f0_unbal_ACLMS)
    % Plotting results for balanced and unbalanced systems as subplots
    figure;
    subplot(2, 1, 1);
    plot(f0_bal_CLMS, 'LineWidth', 1.2);
    hold on;
    plot(f0_bal_ACLMS, 'LineWidth', 1.2);
    yline(50,'k--','LineWidth', 1);
    xlabel('Sample', 'FontSize', 11);
    ylabel('Estimated frequency', 'FontSize', 11);
    title('Balanced System', 'FontSize', 11);
    legend('CLMS', 'ACLMS', 'Theoretical frequency');
    ylim([0 60]);
    xlim([0 1000]);
    grid on;
   

    subplot(2, 1, 2);
    plot(f0_unbal_CLMS, 'LineWidth', 1.2);
    hold on;
    plot(f0_unbal_ACLMS, 'LineWidth', 1.2);
    yline(50,'k--','LineWidth', 1);
    xlabel('Sample', 'FontSize', 11);
    ylabel('Estimated frequency ', 'FontSize', 11);
    title('Unbalanced System', 'FontSize', 11);
    legend('CLMS', 'ACLMS', 'Theoretical frequency');
    ylim([0 60]);
    xlim([0 1000]);
    grid on;
   
end




function [x_hat, error, h, g] = ACLMS_pred_AR(input_x, adapt_gain, filter_order)
    N = length(input_x);  % Total number of samples
    
    % Construct input matrix x_n
    x_n = zeros(filter_order, N);
    for k = 1:filter_order
        x_n(k, k+1:end) = input_x(1:end-k);
    end

    % Initialize variables
    x_hat = zeros(N, 1);
    error = zeros(N, 1);
    h = zeros(filter_order, N+1);  % Weight matrix for h, includes initial zero column
    g = zeros(filter_order, N+1);  % Weight matrix for g, includes initial zero column
    
    % Set initial weights for h to 1
    h(:, 1) = ones(filter_order, 1);
    
    % Main ACLMS loop
    for n = 1:N
        % Compute the output using both h and g weights
        x_hat(n) = h(:, n)' * x_n(:, n) + g(:, n)' * conj(x_n(:, n));
        
        % Calculate the error
        error(n) = input_x(n) - x_hat(n);
        
        % Update weights using the ACLMS algorithm
        h(:, n+1) = h(:, n) + adapt_gain * conj(error(n)) * x_n(:, n);
        g(:, n+1) = g(:, n) + adapt_gain * conj(error(n)) * conj(x_n(:, n));
    end
    
    % Remove the initial zero column from weights
    h = h(:, 2:end);
    g = g(:, 2:end);
end




function [x_hat,error,h] = CLMS_pred_AR(input_x,adapt_gain,filter_order)
    
    N=length(input_x); 
    x_n = zeros(filter_order, N);
    %Obtaining x_n
    for k = 1: filter_order
        x_n(k, :) = [zeros(1, k), input_x(1: N - k)];
    end
    
    %Initialising variables
    error=zeros(1,N);
    h = zeros(filter_order,N+1); %Stores weight time-evolution
    x_hat = zeros(1,N);
    
%     Setting h(0) to 1.
    h(:,1) = ones(filter_order,1);
    
    for n=1:N
        x_hat(n) = h(:,n)'*x_n(:,n);
        %CLMS error
        error(n)=input_x(n)-x_hat(n); %Error calculation
        %CLMS update rule
        h(:,n+1)=h(:,n)+adapt_gain*conj(error(n))*x_n(:,n);
    end
    h = h(:,2:end); %Discarding the first term.
end




function [x_hat, error, h] = CLMS_AR_3b(input_x, adapt_gain, filter_order)
    N = length(input_x);  % Total number of samples
    x_n = zeros(filter_order, N); % Pre-allocate for speed
    
    % Generate lagged versions of the input signal
    for k = 1:filter_order
        x_n(k, k+1:end) = input_x(1:end-k);  % Efficient lagging
    end

    error = zeros(1, N);  % Prediction error initialization
    h = zeros(filter_order, N+1); % Weight matrix including initial column
    x_hat = zeros(1, N);  % Predicted signal initialization
    
    % Adaptive filtering using CLMS
    for n = 1:N
        x_hat(n) = h(:, n)' * x_n(:, n);  % Predicted signal using current weights
        error(n) = input_x(n) - x_hat(n);  % Prediction error
        h(:, n+1) = h(:, n) + adapt_gain * conj(error(n)) * x_n(:, n);  % Weight update
    end
    
    h = h(:, 2:end); % Discard the initial column to align with the error and x_hat
end




function [weights, error] = dft_clms(y, x, leak)
 

    [num_freqs, num_samples] = size(x);
    weights = zeros(num_freqs, num_samples, 'like', 1i); 
    error = zeros(num_samples, 1, 'like', 1i);
    learning_rate = 1;

    for n = 1:num_samples
        error(n) = y(n) - weights(:, n).' * x(:, n);
        if n < num_samples
            weights(:, n+1) = (1 - learning_rate * leak) * weights(:, n) + learning_rate * conj(error(n)) * x(:, n);
        end
    end
end