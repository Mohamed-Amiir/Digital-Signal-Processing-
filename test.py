import numpy as np
def fourier_transform(signal_samples):
    # Calculate the Fourier transform
    n = len(signal_samples)
    frequency = np.fft.fftfreq(n)
    amplitude_spectrum = np.fft.fft2(signal_samples)

    # Calculate the magnitude and phase spectra
    magnitude_spectrum = np.abs(amplitude_spectrum)
    phase_spectrum = np.angle(amplitude_spectrum)

    return frequency, magnitude_spectrum, phase_spectrum

# Input signal
signal_samples = np.array([1, 3, 5, 7, 9, 11, 13, 15])

# Calculate the Fourier transform
frequency, magnitude_spectrum, phase_spectrum = fourier_transform(signal_samples)

# Print the results
print("Frequency | Amplitude Spectrum | Phase Spectrum")
print("---|---|---|")
for i in range(len(frequency)):
    print(f"{frequency[i]:.3f} | {magnitude_spectrum[i]:.3f} | {phase_spectrum[i]:.3f}")
