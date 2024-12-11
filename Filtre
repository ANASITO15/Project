import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Generate a signal with duplicates (sinusoids of different frequencies)
def generate_signal(fs, duration, frequencies, amplitudes):
    """
    Generates a signal composed of multiple sinusoidal components.

    Args:
        fs (int): Sampling frequency (Hz).
        duration (float): Duration of the signal (seconds).
        frequencies (list): Frequencies of the sinusoids.
        amplitudes (list): Amplitudes of the sinusoids.

    Returns:
        t (numpy array): Time vector.
        signal (numpy array): Generated signal.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.zeros_like(t)
    for f, a in zip(frequencies, amplitudes):
        signal += a * np.sin(2 * np.pi * f * t)
    return t, signal

# Low-pass filter design and application
def low_pass_filter(data, cutoff, fs, order=5):
    """
    Applies a low-pass Butterworth filter to the input data.

    Args:
        data (numpy array): Input signal.
        cutoff (float): Cutoff frequency of the filter (Hz).
        fs (int): Sampling frequency (Hz).
        order (int): Order of the Butterworth filter.

    Returns:
        Filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

# User inputs for signal generation
fs = int(input("Enter sampling frequency (Hz): "))
duration = float(input("Enter signal duration (seconds): "))
frequencies = list(map(float, input("Enter frequencies of the sinusoids (comma-separated): ").split(',')))
amplitudes = list(map(float, input("Enter amplitudes of the sinusoids (comma-separated): ").split(',')))

# Generate the signal
t, signal = generate_signal(fs, duration, frequencies, amplitudes)

# User inputs for the low-pass filter
cutoff = float(input("Enter the cutoff frequency for the LPF (Hz): "))
order = int(input("Enter the order of the LPF: "))

# Apply the low-pass filter
filtered_signal = low_pass_filter(signal, cutoff, fs, order)

# Plot the original and filtered signals
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal, label='Original Signal')
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, filtered_signal, label='Filtered Signal', color='orange')
plt.title('Filtered Signal (LPF Applied)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()
