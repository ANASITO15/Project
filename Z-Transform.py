import matplotlib.pyplot as plt
import numpy as np

def z_transform(signal, z_vals):
    """
    Computes the Z-transform of a signal for a range of z values.
    
    Args:
        signal (list or array): Input signal (discrete sequence).
        z_vals (array): Array of z values (complex numbers).
    
    Returns:
        Z-transform values for each z in z_vals.
    """
    n = np.arange(len(signal))  # Sample indices
    return np.array([np.sum(signal * (z**-n)) for z in z_vals])

# Get user input for the signal
signal = input("Enter the signal values as a comma-separated list (e.g., 1,2,3,4): ")
signal = np.array([float(x) for x in signal.split(',')])  # Convert input to a numpy array

# Get user input for z-plane range
z_real_min = float(input("Enter the minimum real part of z: "))
z_real_max = float(input("Enter the maximum real part of z: "))
z_imag_min = float(input("Enter the minimum imaginary part of z: "))
z_imag_max = float(input("Enter the maximum imaginary part of z: "))
resolution = int(input("Enter the resolution (number of points per axis): "))

# Define the z-plane range
real = np.linspace(z_real_min, z_real_max, resolution)
imag = np.linspace(z_imag_min, z_imag_max, resolution)
z_real, z_imag = np.meshgrid(real, imag)
z_vals = z_real + 1j * z_imag  # Complex grid of z-values

# Compute the Z-transform
z_transform_values = np.zeros_like(z_vals, dtype=complex)
for i in range(z_vals.shape[0]):
    for j in range(z_vals.shape[1]):
        z_transform_values[i, j] = z_transform(signal, [z_vals[i, j]])

# Plot the magnitude of the Z-transform
plt.figure(figsize=(10, 6))
plt.contourf(real, imag, np.abs(z_transform_values), levels=50, cmap='viridis')
plt.colorbar(label='|X(z)|')
plt.title('Magnitude of the Z-transform')
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')
plt.grid(True)
plt.show()

# Plot the signal
plt.figure(figsize=(10, 6))
plt.stem(signal, use_line_collection=True)
plt.title('Input Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
