%% Simulation showing equivalence of 2 PSD Definitions

clear all;
clc;
% Scenario 1: Equivalence Holds - Using White Noise

fs = 1000; % Sampling frequency
t = 0:1/fs:1-1/fs; % Time vector
x = randn(size(t)); % Generating white noise

% Plot the white noise signal
figure;
subplot(3,1,1); % Adjusted for an additional plot
plot(t, x);
title('White Noise Signal');
xlabel('Time (seconds)');
ylabel('Amplitude');

% Calculate and plot the Autocorrelation Function (ACF)
[acf, lags] = xcorr(x, 'unbiased'); % 'unbiased' for white noise
subplot(3,1,2); % Adjusted for an additional plot
plot(lags/fs, acf);
title('Autocorrelation Function (ACF) - White Noise');
xlabel('Lag (seconds)');
ylabel('Amplitude');

% PSD calculation and plotting
N = length(x);
f = fs*(-N/2:N/2-1)/N; % Frequency vector for plotting
P1 = fftshift(abs(fft(acf, N))); % FFT of ACF for PSD (Definition 1)
X = fft(x, N);
P2 = (abs(X).^2)/N; % Direct PSD calculation (Definition 2)

subplot(3,1,3);
plot(f, 10*log10(P1), 'DisplayName', 'PSD via Definition 1');
hold on;
plot(f, 10*log10(P2), 'DisplayName', 'PSD via Definition 2');
title('Power Spectral Density (PSD) Comparison - White Noise');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
legend;

%% Simulation where the equivalence does not hold

clear all;
clc;
% Scenario 2: Equivalence Does Not Hold - Persistent Low-Frequency Signal

% Generate a persistent low-frequency sinusoidal signal
fs = 1000; % Sampling frequency
t = 0:1/fs:10-1/fs; % Longer time vector
f = 5; % Frequency of the sinusoid
x = sin(2*pi*f*t); % Sinusoidal signal

% Plot the sinusoidal signal
figure;
subplot(3,1,1); % Adjusted for an additional plot
plot(t, x);
title('Persistent Low-Frequency Sinusoidal Signal');
xlabel('Time (seconds)');
ylabel('Amplitude');

% Calculate and plot the ACF
[acf, lags] = xcorr(x, 'biased');
subplot(3,1,2); % Adjusted for an additional plot
plot(lags/fs, acf);
title('Autocorrelation Function (ACF) - Persistent Signal');
xlabel('Lag (seconds)');
ylabel('Amplitude');

% PSD calculation and plotting
N = length(x);
f = fs*(-N/2:N/2-1)/N; % Adjusting frequency vector
P1 = fftshift(abs(fft(acf, N))); % FFT of ACF for PSD (Definition 1)
X = fft(x, N);
P2 = (abs(X).^2)/N; % Direct PSD calculation (Definition 2)

subplot(3,1,3);
plot(f, 10*log10(P1), 'DisplayName', 'PSD via Definition 1');
hold on;
plot(f, 10*log10(P2), 'DisplayName', 'PSD via Definition 2');
title('Power Spectral Density (PSD) Comparison - Persistent Signal');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
legend;

%% Periodogram based method on sunspot time series

clear all;
clc;
% Load the sunspot time series data included in MATLAB
load sunspot.dat;

% Extract the sunspot data
data = sunspot(:, 2);

% Compute the PSD for raw data
[P_raw, w] = periodogram(data); % PSD for raw data
P_raw(P_raw < 0.001) = 1;


% Compute the PSD for centered data (mean removed)
P_centered = periodogram(data - mean(data)); % PSD for centered data
P_centered(P_centered < 0.01) = 1;

% Compute the PSD for detrended, centered data
P_detrend = periodogram(detrend(data)); % PSD for detrended, centered data
P_detrend(P_detrend < 0.01) = 1;

% Logarithmic transformation of the data
data_log = log10(data + eps);

% Compute the PSD for log-transformed raw data
P_raw_log = periodogram(data_log); % PSD for log-transformed raw data
P_raw_log(P_raw_log < 0.01) = 1;

% Compute the PSD for log-transformed centered data (mean removed)
P_centered_log = periodogram(data_log - mean(data_log)); % PSD for log-transformed centered data
P_centered_log(P_centered_log < 0.01) = 1;

% Compute the PSD for log-transformed detrended, centered data
P_detrend_log = periodogram(detrend(data_log)); % PSD for log-transformed detrended, centered data
P_detrend_log(P_detrend_log < 0.01) = 1;

% Plotting PSD estimates
figure;
subplot(2, 1, 1); hold on;
plot(w, 10*log10(P_raw + eps)); 
plot(w, 10*log10(P_centered + eps));
plot(w, 10*log10(P_detrend + eps));
set(gca,'fontsize', 14);
xlabel('Frequency (cycles/year)'); 
ylabel('Power/frequency (dB/(cycles/year))'); title('PSD Estimates for Original Data');
legend('Raw', 'Centered (Mean Removed)', 'Detrended'); hold off;

subplot(2, 1, 2); hold on;
plot(w, 10*log10(P_raw_log + eps)); 
plot(w, 10*log10(P_centered_log + eps));
plot(w, 10*log10(P_detrend_log + eps));
set(gca,'fontsize', 14);
xlabel('Frequency (cycles/year)'); 
ylabel('Power/frequency (dB/(cycles/year))'); title('PSD Estimates for PSD for Log-transformed  Data');
legend('Raw', 'Centered (Mean Removed)', 'Detrended'); hold off;
%% Periodogram based method on EEG signal

clear all;
clc;
% Load the EEG dataset
load('EEG_Data_Assignment1.mat');

% Define the stimulus frequency range and compute necessary parameters
fs= 1200;
deltaT = 1/fs; % Time step (inverse of sampling frequency)
numSamples = numel(POz); % Total number of EEG samples

% Normalize the EEG signal
normalizedPOz = normalize(POz, 'zscore');

% Compute the Power Spectral Density (PSD) using Welch's method
[ powerSpectralDensity,frequencyVector,] = pwelch(normalizedPOz, numSamples, 0, numSamples, fs);

% Specify window lengths for averaged periodogram analysis
windowLengthsInSeconds = [1, 5, 10];
sampleWindowLengths = windowLengthsInSeconds / deltaT; % Convert window lengths from seconds to samples

% Initialize figure for plotting PSD estimates
figure;
subplot(2, 2, 1);
plot(frequencyVector(1:floor(length(frequencyVector)/6)), 10*log10(powerSpectralDensity(1:floor(length(frequencyVector)/6))));
set(gca, 'FontSize', 14);
xlabel('Frequency (Hz)', 'FontSize', 12);
ylabel('Power/Frequency (dB/Hz)', 'FontSize', 12);
title('Normalised EEG PSD', 'FontSize', 14);

% Loop over different window lengths to compute and plot averaged PSDs
for idx = 1:length(windowLengthsInSeconds)
    [averagedPSD, freqVals] = pwelch(normalizedPOz, sampleWindowLengths(idx), 0, numSamples, fs);
    subplot(2, 2, idx + 1);
    plot(freqVals(1:floor(length(freqVals)/6)), 10*log10(averagedPSD(1:floor(length(freqVals)/6))));
    set(gca, 'FontSize', 14);
    xlabel('Frequency (Hz)', 'FontSize', 12);
    ylabel('Power/Frequency (dB/Hz)', 'FontSize', 12);
    title(['Averaged EEG PSD (Window = ' num2str(windowLengthsInSeconds(idx)) ' s)'], 'FontSize', 14);
end

%% Biased and unbiased ACF estimates of a signal and then use these ACF estimates to compute the corresponding correlogram

clear all;
clc;
% Define the signal length
N = 1024;

% Generate different signals
wgn_signal = randn(N, 1); % White Gaussian Noise
noisy_sin_signal = 10*sin(2*pi*0.05*(1:N))' + 2*randn(N, 1); % Noisy sinusoidal signal
filtered_wgn_signal = filter(5, [0.9 -0.9], wgn_signal); % Filtered White Gaussian Noise

% Calculate correlograms for different signals
[P_wgn_biased, P_wgn_unbiased] = correlogram(wgn_signal, N);
[P_noisy_sin_biased, P_noisy_sin_unbiased] = correlogram(noisy_sin_signal, N);
[P_filtered_wgn_biased, P_filtered_wgn_unbiased] = correlogram(filtered_wgn_signal, N);

% Frequency vector for plotting
f = (0:N-1)*(1/N);

% Plot the correlogram results
figure;

subplot(3,2,1);
plot(f, 10*log10(P_wgn_biased));
title('WGN Biased Correlogram');
xlabel('Normalised Frequency(Hz)');
ylabel('Power/Frequency (dB/Hz)');

subplot(3,2,2);
plot(f, 10*log10(P_wgn_unbiased));
title('WGN Unbiased Correlogram');
xlabel('Normalised Frequency(Hz)');
ylabel('Power/Frequency (dB/Hz)');

subplot(3,2,3);
plot(f, 10*log10(P_noisy_sin_biased));
title('Noisy Sinusoidal Biased Correlogram');
xlabel('Normalised Frequency(Hz)');
ylabel('Power/Frequency (dB/Hz)');

subplot(3,2,4);
plot(f, 10*log10(P_noisy_sin_unbiased));
title('Noisy Sinusoidal Unbiased Correlogram');
xlabel('Normalised Frequency(Hz)');
ylabel('Power/Frequency (dB/Hz)');

subplot(3,2,5);
plot(f, 10*log10(P_filtered_wgn_biased));
title('Filtered WGN Biased Correlogram');
xlabel('Normalised Frequency(Hz)');
ylabel('Power/Frequency (dB/Hz)');

subplot(3,2,6);
plot(f, 10*log10(P_filtered_wgn_unbiased));
title('Filtered WGN Unbiased Correlogram');
xlabel('Normalised Frequency(Hz)');
ylabel('Power/Frequency (dB/Hz)');

%% Plotting the PSD in dB.

clear all;
clc;
N= 64;

wgn100 = randn(1000,N);
noiseSin100 = repmat(4*sin(2*pi*(1:N)*0.4) + sin(2*pi*(1:N)*0.15), 1000,1) + wgn100;

sinB100 = zeros(1000,2*N-1);

for i=1:1000
    sinB100(i,:) = xcorr(noiseSin100(i,:), 'biased');
end

PSDs100 = fftshift(real(fft(ifftshift(sinB100,2)')));

meanPSDs100 = mean(PSDs100,2);
stdPSDs100 = std(PSDs100,0,2);

normf = linspace(0,1,length(meanPSDs100));
figure;
subplot(2, 1, 1);
plot(normf, PSDs100, 'color','cyan');

hold on
plot(normf, meanPSDs100,'b' );
xlabel('Normalised Frequency (Hz)')
ylabel('Power/Frequency')
title('PSD estimates and mean of noisy sinewave')

subplot(2, 1, 2);
plot(normf, stdPSDs100,'r');
xlabel('Normalised Frequency (Hz)')
ylabel('Power/Frequency')

title('STD of noisy sinewave PSD')


PSDs100dB = pow2db(PSDs100);
meanPSDs100dB = pow2db(meanPSDs100);
stdPSDs100dB = pow2db(stdPSDs100);

figure;
subplot(2, 1, 1);
plot(normf, PSDs100dB, 'color','cyan');
hold on
plot(normf, meanPSDs100dB,'b');
xlabel('Normalised Frequency (Hz)')
ylabel('Power/Frequency dB/Hz)')

title('PSD estimates and mean of noisy sinewave (dB)')
subplot(2, 1, 2);
plot(normf, stdPSDs100dB,'r' );
xlabel('Normalised Frequency (Hz)')
ylabel('Power/Frequency (dB/Hz)')

title('STD of noisy sinewave PSD (dB)')


%% Frequency estimation by MUSIC.

clear all;
clc;

% Define parameters
M = 512; % Window size for FFT, choose a value that is a power of 2 for efficiency
numPoints = [20, 50, 100, 300]; % Different lengths of the signal

% Initialize the matrix to store PSD estimates
PSDdB = zeros(length(numPoints), M);

for i = 1:length(numPoints)
    
    n = 0:numPoints(i)-1; % Adjust index for MATLAB
    noise = 0.2/sqrt(2)*(randn(size(n)) + 1j*randn(size(n)));
    x = exp(1j*2*pi*0.3*n) + exp(1j*2*pi*0.32*n) + noise;
    x = [x zeros(1, M - length(x))]; % Zero-padding to match the FFT length
    
    % Calculate FFT and convert to Power Spectral Density in dB/Hz
    X = fft(x, M);
    PSD = (1/(M/2)) * abs(X).^2; % Normalize by M/2 for one-sided PSD in non-dB units
    PSDdB(i,:) = 10 * log10(PSD); % Convert to dB/Hz
    
end

% Normalized frequency axis
normf = linspace(0, 1, M);

% Plotting
figure;
plot(normf, PSDdB(1,:));
hold on;
plot(normf, PSDdB(2,:));
plot(normf, PSDdB(3,:));
plot(normf, PSDdB(4,:));
xlim([0 1]); % Focus on the first half of the normalized frequency range
xlabel('Normalized Frequency (Hz)');
ylabel('Power Spectral Density (dB/Hz)');
legend('n = 20','n = 50','n = 100', 'n = 300');
title('Periodogram of 2 Complex Exponentials for Varying Sample Length (n) ');

%%
% Define the signal parameters
N = 50; % Number of samples
n = 0:N-1; % Time index
f1 = 0.3; % Frequency of the first complex exponential
f2 = 0.32; % Frequency of the second complex exponential
noisePower = 0.2/sqrt(2); % Noise power

% Generate the signal
noise = noisePower * (randn(1, N) + 1j*randn(1, N)); % Complex Gaussian noise
x = exp(1j*2*pi*f1*n) + exp(1j*2*pi*f2*n) + noise; % Noisy signal

% Periodogram analysis with normalized frequency axis
figure;
subplot(2,1,1);
[pxx, freq] = periodogram(x, [], 1024, 1); % Compute periodogram
freq_normalized = freq / max(freq); % Normalize frequency axis to range [0, 1]
plot(freq_normalized, 10*log10(pxx), 'LineWidth', 2); % Plot in dB/Hz
title('Periodogram');
xlabel('Normalized Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
xlim([0 1]);

% MUSIC analysis
subplot(2,1,2);
[X, R] = corrmtx(x, 14, 'mod'); % Generate the correlation matrix
[S, F] = pmusic(R, 2, 1024, 1, 'corr'); % Apply MUSIC algorithm
F_normalized = F / max(F); % Normalize frequency axis to range [0, 1]
plot(F_normalized, S, 'linewidth', 2);
xlim([0 1]); % Adjust as needed based on the expected frequency range
grid on;
title('MUSIC Spectrum Estimate');
xlabel('Normalized Frequency (Hz)');
ylabel('Pseudospectrum');


%% AR Estimation

clear all;
clc;

% change 1000 to 10000 for next part!

% Initialize parameters
N = 10000;
n = 1:N;
sigma = 1;

% Generate white Gaussian noise
x = sigma * randn(size(n));

% Initialize the AR process
x(1) = x(1); % This line is redundant as x(1) is already initialized
x(2) = 2.76 * x(1) + x(2);
x(3) = 2.76 * x(2) - 3.81 * x(1) + x(3);
x(4) = 2.76 * x(3) - 3.81 * x(2) + 2.65 * x(1) + x(4);

% Run the AR process
for i = 5:N
    x(i) = 2.76 * x(i-1) - 3.81 * x(i-2) + 2.65 * x(i-3) - 0.92 * x(i-4) + x(i);
end

% Discard the first 500 samples
x = x(501:end); % chnage this for second part 

% Model orders to estimate
p = 2:14;
h = zeros(length(p), N/2);
aOrig = [2.76, -3.81, 2.65, -0.92];
hOrig = freqz(1, [1, -aOrig], N/2);

% Estimate PSD for different model orders
for i = 1:length(p)
    [a,e] = aryule(x, p(i));
    [h(i,:), ~] = freqz(sqrt(e), a, N/2);
end

% Normalized frequency vector
normf = linspace(0, 1, length(hOrig));

% Plot the results
figure;
plot(normf, pow2db(abs(h(4,:)').^2));
hold on;
plot(normf, pow2db(abs(h(8,:)').^2));
plot(normf, pow2db(abs(h(11,:)').^2));
plot(normf, pow2db(abs(h(13,:)').^2));
plot(normf, pow2db(abs(hOrig).^2));
xlim([0, 1]);
xlabel('Normalized Frequency (Hz)');
ylabel('Power Spectral Density (dB/Hz)');
legend( 'AR(4)','AR(8)', 'AR(11)', 'AR(13)', 'Original');
title('PSD Estimation for Varying Orders (p)');


%% Real World Signals: Respiratory Sinus Arrhythmia from RR-Intervals
%% (a) and (b)

clear all;
clc;

% TRIAL 1:
load('RRI1.mat');
load('RRI2.mat');
load('RRI3.mat');
% Standard periodogram for the first RRI data set
[pxx1, f1] = periodogram(xRRI1, [], [], fsRRI1);
[pxx2, f2] = periodogram(xRRI2, [], [], fsRRI2);
[pxx3, f3] = periodogram(xRRI3, [], [], fsRRI3);

% Plot
figure;
subplot(1,2,1);
plot(f1, 10*log10(pxx1));
xlabel('Normalised Frequency (Hz)');
ylabel('Power Spectral Density (dB/Hz)');
title('Standard Periodogram of First RRI Data');
xlim([0, 1]);
% Example window length in samples (assuming fsRRI1 is in Hz)
windowLengthSamples50 = 50* fsRRI1;% 50 seconds window
windowLengthSamples150 = 150* fsRRI1;
windowLengthSamples200 = 200* fsRRI1;
[windowpxx1, f1] = pwelch(xRRI1, windowLengthSamples50, [], [], fsRRI1);
subplot(1,2,2);
plot(f1, 10*log10(windowpxx1));
hold on;
[windowpxx1, f1] = pwelch(xRRI1, windowLengthSamples150, [], [], fsRRI1);
plot(f1, 10*log10(windowpxx1));
hold on;
[windowpxx1, f1] = pwelch(xRRI1, windowLengthSamples200, [], [], fsRRI1);
plot(f1, 10*log10(windowpxx1));
legend('Window Length = 50','Window Length = 150','Window Length = 200')
xlim([0, 1]);
xlabel('Normalised Frequency (Hz)');
ylabel('Power Spectral Density (dB/Hz)');
title('Averaged Periodogram of First RRI Data for Different Window Lengths');



% TRIAL 2:
% Plot
figure;
subplot(1,2,1);
plot(f2, 10*log10(pxx2));
xlabel('Normalised Frequency (Hz)');
ylabel('Power Spectral Density (dB/Hz)');
title('Standard Periodogram of Second RRI Data');
xlim([0, 1]);
% Example window length in samples (assuming fsRRI1 is in Hz)
windowLengthSamples50 = 50* fsRRI2;% 50 seconds window
windowLengthSamples150 = 150* fsRRI2;
windowLengthSamples200 = 200* fsRRI2;
[windowpxx2, f2] = pwelch(xRRI2, windowLengthSamples50, [], [], fsRRI2);
subplot(1,2,2);
plot(f2, 10*log10(windowpxx2));
hold on;
[windowpxx2, f2] = pwelch(xRRI2, windowLengthSamples150, [], [], fsRRI2);
plot(f2, 10*log10(windowpxx2));
hold on;
[windowpxx2, f2] = pwelch(xRRI2, windowLengthSamples200, [], [], fsRRI2);
plot(f2, 10*log10(windowpxx2));
legend('Window Length = 50','Window Length = 150','Window Length = 200')
xlim([0, 1]);
xlabel('Normalised Frequency (Hz)');
ylabel('Power Spectral Density (dB/Hz)');
title('Averaged Periodogram of Second RRI Data for Different Window Lengths');

% TRIAL 3:
% Plot
figure;
subplot(1,2,1);
plot(f3, 10*log10(pxx3));
xlabel('Normalised Frequency (Hz)');
ylabel('Power Spectral Density (dB/Hz)');
title('Standard Periodogram of Third RRI Data');
xlim([0, 1]);
% Example window length in samples (assuming fsRRI1 is in Hz)
windowLengthSamples50 = 50* fsRRI3;% 50 seconds window
windowLengthSamples150 = 150* fsRRI3;
windowLengthSamples200 = 200* fsRRI3;
[windowpxx3, f3] = pwelch(xRRI3, windowLengthSamples50, [], [], fsRRI3);
subplot(1,2,2);
plot(f3, 10*log10(windowpxx3));
hold on;
[windowpxx3, f3] = pwelch(xRRI3, windowLengthSamples150, [], [], fsRRI3);
plot(f1, 10*log10(windowpxx1));
hold on;
[windowpxx3, f3] = pwelch(xRRI3, windowLengthSamples200, [], [], fsRRI3);
plot(f3, 10*log10(windowpxx3));
legend('Window Length = 50','Window Length = 150','Window Length = 200')
xlim([0, 1]);
xlabel('Normalised Frequency (Hz)');
ylabel('Power Spectral Density (dB/Hz)');
title('Averaged Periodogram of Third RRI Data for Different Window Lengths');


%% (c)

clear all;
clc;

load('RRI1.mat');
load('RRI2.mat');
load('RRI3.mat');
% Standard periodogram for the first RRI data set
[pxx1, f1] = periodogram(xRRI1, [], [], fsRRI1);
[pxx2, f2] = periodogram(xRRI2, [], [], fsRRI2);
[pxx3, f3] = periodogram(xRRI3, [], [], fsRRI3);
fs = fsRRI1;
p = [1 4 8 12 20 33 40];

for i = 1:length(p)   
    % Straight forward implementation    
    [a, e] = aryule(xRRI1, p(i));
    [A, w1] = freqz(e.^(1/2), a, length(xRRI1), fs);
    pxx1AR(i, :) = abs(A).^2;     
end

for i = 1:length(p)   
    % Straight forward implementation    
    [a, e] = aryule(xRRI2, p(i));
    [A, w2] = freqz(e.^(1/2), a, length(xRRI2), fs);
    pxx2AR(i, :) = abs(A).^2;     
end

for i = 1:length(p)   
    % Straight forward implementation    
    [a, e] = aryule(xRRI3, p(i));
    [A, w3] = freqz(e.^(1/2), a, length(xRRI3), fs);
    pxx3AR(i, :) = abs(A).^2;     
end


figure;
hold on;
plot(f1, 10*log10(pxx1));
plot(w1, 10*log10(pxx1AR));
xlabel('Normalised Frequency (Hz)');
ylabel('Power Spectral Density (dB/Hz)');
xlim([0, 1]);
legend('Original','p=1','p=4','p=8','p=12','p=20','p=33', 'p=40')
title('AR Spectrum Estimation for Trial 1')

figure;
hold on;
plot(f2, 10*log10(pxx2));
plot(w2, 10*log10(pxx2AR));
xlabel('Normalised Frequency (Hz)');
ylabel('Power Spectral Density (dB/Hz)');
xlim([0, 1]);
legend('Original','p=1','p=4','p=8','p=12','p=20','p=33', 'p=40')
title('AR Spectrum Estimation for Trial 2')

figure;
hold on;
plot(f3, 10*log10(pxx3));
plot(w3, 10*log10(pxx3AR));
xlabel('Normalised Frequency (Hz)');
ylabel('Power Spectral Density (dB/Hz)');
xlim([0, 1]);
legend('Original','p=1','p=4','p=8','p=12','p=20','p=33', 'p=40')
title('AR Spectrum Estimation for Trial 3')


%% Robust Regression
%% (a)

clear all;
clc;

load('PCAPCR.mat');
% svd
[U, S, V] = svd(X, 'econ'); % For the original input matrix X
[U_noise, S_noise, V_noise] = svd(Xnoise, 'econ'); % For the noise corrupted input matrix Xnoise

%plot singular values 
figure;
stem(diag(S), 'filled', 'LineWidth', 2);
hold on;
stem(diag(S_noise), 'LineWidth', 2);
legend('X', 'X_{noise}');
title('Singular Values of X and X_{noise}');
xlabel('Index');
ylabel('Singular Value');

% calculate and plot square error
square_errors = (diag(S) - diag(S_noise)).^2;
figure;
stem(square_errors, 'filled', 'LineWidth', 2);
title('Square Error between Singular Values of X and X_{noise}');
xlabel('Index');
ylabel('Square Error');



%% (B)

clear all;
clc;

load('PCAPCR.mat');
svdX = svd(X);
svdXN = svd(Xnoise);
for rank=1:length(svdX)
    [U,S,V] = svd(Xnoise);
    
    U = U(:,1:rank);
    S = S(1:rank,1:rank);
    V = V(:,1:rank);

    Xreconstructed = U*S*V';

    e1(rank) = sum(sum((X-Xreconstructed).^2));
    e2(rank) = sum(sum((Xreconstructed-Xnoise).^2));
end

figure
plot(e1)
hold on
plot(e2)
xlabel('Rank')
ylabel('Error')

legend('X - X_{reconstructed}','X_{reconstructed} - X_{noise}')
title('Squared Error as a Function of Reconstruction Rank')

%% (c) & (d)



clear all;
clc;

load('PCAPCR.mat');
Bols = (Xnoise'*Xnoise)\Xnoise'*Y;
Yols = Xnoise*Bols;

rank = 3;
[U,S,V] = svd(Xnoise);
U = U(:,1:rank);
S = S(1:rank,1:rank);
V = V(:,1:rank);
BPCR = V/S*U'*Y;
Ypcr = Xnoise*BPCR;

eOLS = sum(sum((Y-Yols).^2))/numel(Y);
ePCR = sum(sum((Y-Ypcr).^2))/numel(Y);

YolsT = Xtest*Bols;
YpcrT = Xtest*BPCR;

eOLST = sum(sum((Ytest-YolsT).^2))/numel(Ytest);
ePCRT = sum(sum((Ytest-YpcrT).^2))/numel(Ytest);

[YhatO, YO] = regval(Bols);
[YhatP, YP] = regval(BPCR);

eO = sum(sum((YO-YhatO).^2))/numel(YO);
eP = sum(sum((YP-YhatP).^2))/numel(YP);
%% 
% 

function [Pxx_biased, Pxx_unbiased] = correlogram(x, N)
    % Preallocate arrays for ACF
    r_biased = zeros(2*N-1, 1);
    r_unbiased = zeros(2*N-1, 1);
    lag = -N+1:N-1;
    
    % Calculate the biased ACF
    for k = lag
        if k >= 0
            r_biased(k+N) = sum(x(1:N-k) .* conj(x(k+1:N))) / N;
        else
            r_biased(k+N) = conj(sum(x(1-k:N) .* conj(x(1:N+k)))) / N;
        end
    end
    
    % Calculate the unbiased ACF
    for k = lag
        if k >= 0
            r_unbiased(k+N) = sum(x(1:N-k) .* conj(x(k+1:N))) / (N-k);
        else
            r_unbiased(k+N) = conj(sum(x(1-k:N) .* conj(x(1:N+k)))) / (N+k);
        end
    end
    
    % Compute the correlogram using the biased and unbiased ACF
    Pxx_biased = abs(fft(r_biased));
    Pxx_unbiased = abs(fft(r_unbiased));
    
    % Return only the positive frequencies
    Pxx_biased = Pxx_biased(1:N);
    Pxx_unbiased = Pxx_unbiased(1:N);
end